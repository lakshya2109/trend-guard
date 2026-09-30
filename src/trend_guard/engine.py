"""Live/paper orchestration: one 60-second loop plus the hourly cycle. BUILD_SPEC.md Section 10.3.

The engine owns all I/O (Roostoo, Binance, SQLite, logs, the clock). Every decision it acts on comes
from the pure functions in strategy/ and risk/ via strategy.pipeline.decide and
execution.order_manager.plan_orders -- the same calls the backtest makes.

Loop (every 60 s):
  HALT file present -> cancel our working orders, do nothing else.
  refresh tickers -> equity snapshot -> protective checks (trailing stops, shock filter)
  -> service working limit orders (reprice / escalate)
  every 5 min: reconcile + clock sync
  at minute == rebalance_minute (once per hour): hourly cycle
  00:05 UTC: universe refresh; 00:00 UTC: reset the daily P&L anchor
  18:00 UTC: warn if no fill today (never fakes a trade)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from trend_guard.clock import Clock
from trend_guard.config import EnvConfig, StrategyConfig, SymbolsConfig, config_hash, load_events
from trend_guard.data import binance_live
from trend_guard.data.universe import compute_eligible_universe
from trend_guard.execution.broker import Broker, Exposure
from trend_guard.execution.order_manager import OrderExecutor, PlannedOrder, plan_orders
from trend_guard.execution.sim_broker import SimBroker
from trend_guard.risk.protective import shock_filter
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.errors import ShortsDisabledError
from trend_guard.roostoo.models import Ticker
from trend_guard.roostoo.rules import TradingRule, parse_trading_rules
from trend_guard.state.store import Store
from trend_guard.strategy.pipeline import (
    CRYPTO,
    CRYPTO_BETA,
    OverlayInputs,
    StopState,
    asset_signal,
    atr_series,
    bucket_for,
    classify,
    decide,
    donchian_mid,
    ewma_cov_annualised,
    median_lookback,
    ratchet,
    stop_level,
    warmup_bars,
)
from trend_guard.strategy.regime import NEUTRAL
from trend_guard.strategy.signals import ANN_H, ewma_vol

log = logging.getLogger("engine")
dlog = logging.getLogger("decisions")

LOOP_S = 60
RECONCILE_EVERY_MIN = 5
UNIVERSE_REFRESH = (0, 5)  # 00:05 UTC
NO_FILL_WARN_HOUR = 18
COV_BARS = 720  # 30 days of hourly returns is plenty for a 72h-half-life EWMA
FUNDING_Z_LOOKBACK = 90  # 30 days of 8-hourly funding prints
SHOCK_WINDOW = timedelta(minutes=15)
FREEZE = timedelta(minutes=60)
BASIS_LIMIT = 0.005  # Section 5.5: median |basis| above 0.5% over 24h -> distrust Binance bars


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def binance_symbol(pair: str, symbols: SymbolsConfig) -> str:
    return symbols.overrides.get(pair) or pair.split("/")[0] + "USDT"


class Engine:
    def __init__(
        self,
        cfg: StrategyConfig,
        symbols: SymbolsConfig,
        env: EnvConfig,
        client: RoostooClient,
        broker: Broker,
        store: Store,
        clock: Clock,
        halt_path: Path,
        config_path: str = "config/strategy.yaml",
        bars_dir: Path = Path("data/bars/1h"),
        fills_dir: Path = Path("logs/fills"),
    ):
        self.cfg, self.symbols, self.env = cfg, symbols, env
        self.client, self.broker, self.store, self.clock = client, broker, store, clock
        self.halt_path = halt_path
        self.config_path = config_path
        self.bars_dir = bars_dir
        self.fills_dir = fills_dir
        self.events = load_events()
        self.commit = git_commit()  # every fill row and decision carries this (Screen 1 traceability)
        self.rules: dict[str, TradingRule] = {}
        self.exchange_info: dict = {}
        self.universe: set[str] = set()
        self.spread_streak: dict[str, int] = {}
        self.spread_hist: dict[str, deque] = {}
        self.basis_hist: dict[str, deque] = {}
        self.basis_excluded: set[str] = set()
        self.btc_ticks: deque = deque(maxlen=60)
        self.bars: dict[str, pd.DataFrame] = {}  # latest closed hourly bars per pair
        self.shorts_enabled = store.kv_get("shorts_disabled") != "1"
        self.fapi_ok = False
        self.crowding_down_day: str | None = None  # circuit breaker: skip crowding fetches until tomorrow
        self.regime = store.kv_get("regime") or NEUTRAL
        self.last_cycle_hour: str | None = None
        self.last_reconcile: datetime | None = None
        self.last_universe_day: str | None = None
        self.last_warn_day: str | None = None
        self.freeze_longs_until = self._kv_time("freeze_longs_until")
        self.freeze_shorts_until = self._kv_time("freeze_shorts_until")
        self.executor = OrderExecutor(
            broker=broker,
            exec_cfg=cfg.execution,
            clock=clock,
            rules=self.rules,
            on_order=self._on_order,
            on_fill=self._on_fill,
        )

    # --- small persistence helpers ------------------------------------------------------------
    def _kv_time(self, key: str) -> datetime | None:
        v = self.store.kv_get(key)
        return datetime.fromisoformat(v) if v else None

    def _on_order(self, record: dict) -> None:
        self.store.insert_order(record)

    def _on_fill(
        self, pair: str, side: str, qty: Decimal, price: Decimal, fee: Decimal, role: str, reason: str
    ) -> None:
        now = self.clock.now()
        self.store.kv_set(f"last_trade:{pair}", now.isoformat())
        self.store.kv_set("last_fill_day", now.strftime("%Y-%m-%d"))
        self.fills_dir.mkdir(parents=True, exist_ok=True)
        path = self.fills_dir / f"{now:%Y-%m-%d}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "pair", "side", "qty", "price", "fee", "role", "reason", "commit"])
            w.writerow([now.isoformat(), pair, side, qty, price, fee, role, reason, self.commit[:12]])

    def _last_trades(self) -> dict[str, datetime]:
        out = {}
        for p in self.universe | set(self.store.all_positions_meta()):
            v = self.store.kv_get(f"last_trade:{p}")
            if v:
                out[p] = datetime.fromisoformat(v)
        return out

    # --- startup ------------------------------------------------------------------------------
    def startup(self) -> None:
        log.info(
            "startup",
            extra={
                "commit": self.commit,
                "config_hash": config_hash(self.config_path),
                "mode": self.env.bot_mode,
                "risk_scale": self.cfg.competition.risk_scale,
            },
        )
        self.client.sync_clock()
        self.exchange_info = self.client.exchange_info()
        self.rules.update(parse_trading_rules(self.exchange_info))
        self.fapi_ok = binance_live.probe_fapi_access()
        if not self.fapi_ok:
            log.warning("fapi.binance.com unreachable; crowding overlay uses Bybit or is neutral")
        tickers = self.client.get_ticker()
        if isinstance(self.broker, SimBroker):
            self.broker.mark_to_market(tickers)
        self._refresh_universe(tickers, force=True)
        exp = self.broker.get_exposure()
        if self.store.kv_get("start_equity") is None:
            self.store.kv_set("start_equity", exp.equity)
            self.store.kv_set("start_ts", self.clock.now().isoformat())
            self.store.kv_set("peak", exp.equity)
            self.store.kv_set("day_anchor", exp.equity)
        self._load_bars()

    # --- universe -----------------------------------------------------------------------------
    def _history_days(self, pair: str) -> int:
        f = self.bars_dir / f"{binance_symbol(pair, self.symbols)}.parquet"
        if f.exists():
            idx = pd.read_parquet(f, columns=["close"]).index
            return int((idx[-1] - idx[0]).days) if len(idx) else 0
        return 0

    def _refresh_universe(self, tickers: dict[str, Ticker], force: bool = False) -> None:
        pairs = self.exchange_info.get("TradePairs", {})
        hist = {p: self._history_days(p) for p in pairs}
        # before 6h of recorded spreads exist, use the current spread as the estimate
        med = {
            p: float(np.median(self.spread_hist[p])) if self.spread_hist.get(p) else t.spread_pct
            for p, t in tickers.items()
        }
        new, self.spread_streak = compute_eligible_universe(
            self.exchange_info,
            tickers,
            hist,
            med,
            self.symbols,
            self.cfg.universe,
            self.universe,
            self.spread_streak,
        )
        if new != self.universe or force:
            log.info(
                "universe",
                extra={
                    "eligible": sorted(new),
                    "added": sorted(new - self.universe),
                    "dropped": sorted(self.universe - new),
                },
            )
        self.universe = new

    # --- bars ---------------------------------------------------------------------------------
    def _load_bars(self) -> None:
        """Closed hourly bars per eligible pair: local parquet history + the latest 1000 from Binance."""
        now = self.clock.now()
        need = warmup_bars(self.cfg)
        for pair in sorted(self.universe):
            sym = binance_symbol(pair, self.symbols)
            recent = binance_live.fetch_recent_klines_1h(sym, now)
            f = self.bars_dir / f"{sym}.parquet"
            hist = pd.read_parquet(f) if f.exists() and need > len(recent) else pd.DataFrame()
            df = pd.concat([hist, recent]) if len(hist) else recent
            if df.empty:
                log.warning("no bars", extra={"pair": pair})
                continue
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df, stale = binance_live.check_continuity(df[["open", "high", "low", "close"]].astype(float))
            if stale and len(df) and df.index[-1] < now - timedelta(hours=3):
                log.warning("stale bars; excluding", extra={"pair": pair})
                continue
            self.bars[pair] = df.tail(need + COV_BARS)

    # --- equity -------------------------------------------------------------------------------
    def _snapshot(self, exp: Exposure) -> tuple[float, float]:
        now = self.clock.now()
        peak = max(Decimal(self.store.kv_get("peak") or exp.equity), exp.equity)
        self.store.kv_set("peak", peak)
        if now.hour == 0 and self.store.kv_get("day_anchor_day") != now.strftime("%Y-%m-%d"):
            self.store.kv_set("day_anchor", exp.equity)
            self.store.kv_set("day_anchor_day", now.strftime("%Y-%m-%d"))
        longs = sum((exp.long_notional(p) for p in exp.long_qty), Decimal(0))
        shorts = sum(exp.short_collateral.values(), Decimal(0))
        dd = 1 - exp.equity / peak if peak > 0 else Decimal(0)
        self.store.record_equity(
            now.replace(second=0, microsecond=0).isoformat(), exp.equity, exp.cash, longs, shorts, peak, dd
        )
        return float(exp.equity), float(peak)

    # --- protective loop (60 s) ----------------------------------------------------------------
    def _protective(self, exp: Exposure, tickers: dict[str, Ticker]) -> None:
        now = self.clock.now()
        pcfg = self.cfg.protective
        urgent: list[PlannedOrder] = []
        meta = self.store.all_positions_meta()
        held = {p: "LONG" for p, q in exp.long_qty.items() if q > 0}
        held.update({p: "SHORT" for p, c in exp.short_collateral.items() if c > 0})

        for pair, side in held.items():
            t = tickers.get(pair)
            if t is None or not t.mid:
                continue
            price = float(t.mid)
            m = meta.get(pair)
            if m is None or m["side"] != side:  # new position (or flipped): start tracking it
                self.store.upsert_positions_meta(
                    pair, side, now.isoformat(), Decimal(str(price)), None, m["cooldown_until"] if m else None
                )
                continue
            st = StopState(side, float(m["best_price"]), float(m["stop_ref"]) if m["stop_ref"] else None)
            bars = self.bars.get(pair)
            a = (
                float(atr_series(bars, pcfg.atr_bar_h, pcfg.atr_window).iloc[-1])
                if bars is not None
                else math.nan
            )
            mid = (
                float(donchian_mid(bars["close"], median_lookback(self.cfg.signal.lookbacks_h)).iloc[-1])
                if bars is not None
                else None
            )
            if not math.isnan(a):
                level = stop_level(st, a, pcfg.atr_mult)
                hit = price < level if side == "LONG" else price > level
                if hit:
                    until = (now + timedelta(hours=pcfg.cooldown_h)).isoformat()
                    self.store.upsert_positions_meta(pair, side, m["opened_utc"], None, None, until)
                    self.store.kv_set(f"cooldown:{pair}", f"{side}|{until}")  # no same-side re-entry
                    if side == "LONG":
                        urgent.append(
                            PlannedOrder(
                                pair,
                                "SELL",
                                True,
                                True,
                                qty=exp.long_qty[pair],
                                reason=f"trailing_stop level={level:.6g}",
                            )
                        )
                    else:
                        urgent.append(
                            PlannedOrder(
                                pair,
                                "SHORT_CLOSE",
                                True,
                                True,
                                close_pct=Decimal(100),
                                reason=f"trailing_stop level={level:.6g}",
                            )
                        )
                    continue
            st = ratchet(st, price, price, price, mid)
            self.store.upsert_positions_meta(
                pair,
                side,
                m["opened_utc"],
                Decimal(str(st.best_price)),
                Decimal(str(st.stop_ref)) if st.stop_ref is not None else None,
                m["cooldown_until"],
            )
        # positions that no longer exist keep only their cooldown
        for pair, m in meta.items():
            if pair not in held and m["side"] is not None:
                self.store.upsert_positions_meta(pair, None, None, None, None, m["cooldown_until"])

        # shock / squeeze filter on the 15-minute BTC move from recorded ticks
        btc = tickers.get("BTC/USD")
        if btc is not None and btc.mid:
            self.btc_ticks.append((now, float(btc.mid)))
            past = [p for ts, p in self.btc_ticks if now - ts >= SHOCK_WINDOW]
            bb = self.bars.get("BTC/USD")
            if past and bb is not None and len(bb) > 72:
                r15 = math.log(float(btc.mid) / past[-1])
                sigma_h = float(
                    ewma_vol(np.log(bb["close"] / bb["close"].shift(1)), self.cfg.vol.halflife_h).iloc[-1]
                )
                r72 = float(bb["close"].iloc[-1] / bb["close"].iloc[-73] - 1.0)
                act = shock_filter(r15, sigma_h, pcfg, r72)
                if act["cut_longs_pct"] > 0:
                    self.freeze_longs_until = now + FREEZE
                    self.store.kv_set("freeze_longs_until", self.freeze_longs_until.isoformat())
                    for p, q in exp.long_qty.items():
                        if classify(p, self.symbols) in (CRYPTO, CRYPTO_BETA) and q > 0:
                            urgent.append(
                                PlannedOrder(
                                    p,
                                    "SELL",
                                    True,
                                    True,
                                    qty=self.rules[p].clamp_qty(q / 2) if p in self.rules else q / 2,
                                    reason=f"shock r15={r15:.4f}",
                                )
                            )
                if act["cover_shorts_pct"] > 0:
                    self.freeze_shorts_until = now + FREEZE
                    self.store.kv_set("freeze_shorts_until", self.freeze_shorts_until.isoformat())
                    for p in exp.short_collateral:
                        urgent.append(
                            PlannedOrder(
                                p,
                                "SHORT_CLOSE",
                                True,
                                True,
                                close_pct=Decimal(50),
                                reason=f"squeeze r15={r15:.4f}",
                            )
                        )
        if urgent:
            log.warning("protective exits", extra={"orders": [(o.pair, o.action, o.reason) for o in urgent]})
            self._execute(urgent, f"protective-{now:%Y%m%dT%H%M}")

    # --- hourly cycle ------------------------------------------------------------------------------
    def _crowding_inputs(self, pairs: list[str]) -> tuple[dict, dict]:
        fz: dict[str, float | None] = {}
        oi: dict[str, float | None] = {}
        today = self.clock.now().strftime("%Y-%m-%d")
        if not self.cfg.risk.crowding.enabled or self.crowding_down_day == today:
            return fz, oi  # neutral multipliers (1.0); overlays report "crowding data unavailable"
        tried = 0
        for p in pairs:
            if classify(p, self.symbols) != CRYPTO:
                continue
            sym = binance_symbol(p, self.symbols)
            tried += 1
            f = binance_live.fetch_funding(sym, self.fapi_ok, limit=FUNDING_Z_LOOKBACK)
            if len(f) >= 10:
                r = f["funding_rate"]
                sd = float(r.std())
                fz[p] = float((r.iloc[-1] - r.mean()) / sd) if sd > 0 else None
            o = binance_live.fetch_open_interest(sym, self.fapi_ok, period="1h", limit=30)
            if len(o) >= 25:
                oi[p] = float(o["open_interest"].iloc[-1] / o["open_interest"].iloc[-25] - 1.0)
            if tried == 1 and p not in fz and p not in oi:
                # first asset got nothing from any source: the venues are blocked from this host
                self.crowding_down_day = today
                self.fapi_ok = binance_live.probe_fapi_access()  # re-probe tomorrow via this flag
                log.warning("crowding data unavailable from this host; neutral until next UTC day")
                break
        return fz, oi

    def _log_basis(self, tickers: dict[str, Ticker]) -> None:
        """Section 5.5: log Roostoo vs Binance basis hourly; distrust Binance bars above 0.5%."""
        basis = {}
        for p, df in self.bars.items():
            t = tickers.get(p)
            if t is None or not t.mid or df.empty:
                continue
            b = float(t.mid) / float(df["close"].iloc[-1]) - 1.0
            basis[p] = b
            h = self.basis_hist.setdefault(p, deque(maxlen=24))
            h.append(abs(b))
            if len(h) == h.maxlen and float(np.median(h)) > BASIS_LIMIT and p not in self.basis_excluded:
                # no deep Roostoo-bar history exists yet to replace the Binance signal, so the asset
                # is excluded (weight 0) rather than traded on a feed that disagrees with the venue
                self.basis_excluded.add(p)
                log.warning(
                    "basis too wide; excluding from signals",
                    extra={"pair": p, "median_abs": float(np.median(h))},
                )
        dlog.info("basis", extra={"basis": basis})

    def hourly_cycle(self) -> None:
        now = self.clock.now()
        cycle_id = now.strftime("%Y%m%dT%H")
        self._load_bars()
        tickers = self.client.get_ticker()
        self._log_basis(tickers)
        exp = self.broker.get_exposure()
        equity = float(exp.equity)

        pairs = sorted(p for p in self.universe if p in self.bars and p not in self.basis_excluded)
        asset_class = {p: classify(p, self.symbols) for p in pairs}
        bucket_of = {p: bucket_for(p, asset_class[p], self.cfg.universe) for p in pairs}
        closes = {p: self.bars[p]["close"] for p in pairs}
        signals = {p: float(asset_signal(c, self.cfg).iloc[-1]) for p, c in closes.items()}
        log_ret = pd.DataFrame({p: np.log(c / c.shift(1)) for p, c in closes.items()}).tail(COV_BARS)
        sigma_ann = {
            p: float(ewma_vol(log_ret[p].dropna(), self.cfg.vol.halflife_h).iloc[-1] * math.sqrt(ANN_H))
            for p in pairs
            if log_ret[p].notna().any()
        }
        cov = ewma_cov_annualised(log_ret.fillna(0.0), self.cfg.portfolio.cov_halflife_h)

        # market index: top index_size crypto by Roostoo 24h traded value (same construction as backtest)
        crypto = [p for p in pairs if asset_class[p] == CRYPTO]
        top = sorted(crypto, key=lambda p: tickers[p].unit_trade_value if p in tickers else 0, reverse=True)
        idx_ret = pd.DataFrame(
            {p: np.log(closes[p] / closes[p].shift(1)) for p in top[: self.cfg.regime.index_size]}
        )
        idx_price = np.exp(idx_ret.mean(axis=1).fillna(0.0).cumsum())
        mkt = float(asset_signal(idx_price, self.cfg).iloc[-1]) if len(idx_price) else float("nan")

        fz, oi = self._crowding_inputs(pairs)
        start_eq = float(self.store.kv_get("start_equity") or equity)
        comp_start = datetime.fromisoformat(self.cfg.competition.start_utc.replace("Z", "+00:00"))
        cooldown_long, cooldown_short = set(), set()
        for p in pairs:
            v = self.store.kv_get(f"cooldown:{p}")
            if v:
                side, until = v.split("|", 1)
                if datetime.fromisoformat(until) > now:
                    (cooldown_long if side == "LONG" else cooldown_short).add(p)
        current_w = {p: float(exp.long_notional(p)) / equity for p in exp.long_qty} if equity else {}
        for p, c in exp.short_collateral.items():
            current_w[p] = current_w.get(p, 0.0) - float(c) / equity
        ov = OverlayInputs(
            now=now,
            equity=equity,
            peak_equity=float(self.store.kv_get("peak") or equity),
            pnl_today_pct=equity / float(self.store.kv_get("day_anchor") or equity) - 1.0,
            competition_day=max(1, (now - comp_start).days + 1),
            cumulative_return=equity / start_eq - 1.0,
            events=self.events,
            funding_z=fz,
            oi_chg_24h=oi,
            ret_72h={
                p: float(closes[p].iloc[-1] / closes[p].iloc[-73] - 1.0) for p in pairs if len(closes[p]) > 72
            },
            freeze_longs=bool(self.freeze_longs_until and now < self.freeze_longs_until),
            freeze_shorts=bool(self.freeze_shorts_until and now < self.freeze_shorts_until),
            cooldown_long=cooldown_long,
            cooldown_short=cooldown_short,
        )
        dec = decide(
            signals,
            mkt,
            self.regime,
            sigma_ann,
            cov,
            asset_class,
            bucket_of,
            current_w,
            self.shorts_enabled,
            ov,
            self.cfg,
        )
        self.regime = dec.regime
        self.store.kv_set("regime", dec.regime)

        # held assets that dropped out of the universe get target 0 through plan_orders
        plans = plan_orders(dec.weights_final, exp, self.cfg.execution, now, self._last_trades(), self.rules)
        payload = dec.to_json() | {
            "cycle_id": cycle_id,
            "equity": equity,
            "commit": self.commit[:12],
            "targets_notional": {p: w * equity for p, w in dec.weights_final.items()},
            "orders_planned": [
                {
                    "pair": o.pair,
                    "action": o.action,
                    "qty": str(o.qty),
                    "collateral": str(o.collateral),
                    "close_pct": str(o.close_pct),
                    "notional": str(o.notional),
                    "reason": o.reason,
                }
                for o in plans
            ],
        }
        self.store.record_decision(cycle_id, now.isoformat(), dec.regime, payload)
        dlog.info("cycle", extra=json.loads(json.dumps(payload, default=str)))
        self._execute(plans, cycle_id)

    def _execute(self, plans: list[PlannedOrder], cycle_id: str) -> None:
        self.executor.shorts_enabled = self.shorts_enabled
        try:
            self.executor.execute(plans, cycle_id)
        except ShortsDisabledError as e:
            # Section 4.2: disable the short sleeve for the rest of the session, permanently
            self.shorts_enabled = False
            self.store.kv_set("shorts_disabled", "1")
            log.error("shorts disabled by exchange; short sleeve off for the session", extra={"err": str(e)})
            self.executor.execute([p for p in plans if p.action in ("BUY", "SELL")], cycle_id)

    # --- the loop ---------------------------------------------------------------------------------
    def tick(self) -> None:
        now = self.clock.now()
        if self.halt_path.exists():
            self.executor.cancel_all_working()
            log.warning("HALT file present; idle", extra={"path": str(self.halt_path)})
            return

        tickers = self.client.get_ticker()
        for p, t in tickers.items():
            self.spread_hist.setdefault(p, deque(maxlen=6 * 60)).append(t.spread_pct)  # ~6h at 60 s
        if isinstance(self.broker, SimBroker):
            self.broker.mark_to_market(tickers)
        exp = self.broker.get_exposure()
        self._snapshot(exp)
        self._protective(exp, tickers)
        self.executor.service()

        if self.last_reconcile is None or now - self.last_reconcile >= timedelta(minutes=RECONCILE_EVERY_MIN):
            self.client.sync_clock()
            self.last_reconcile = now  # exposure is re-read from the exchange every tick already

        day = now.strftime("%Y-%m-%d")
        if (now.hour, now.minute) >= UNIVERSE_REFRESH and self.last_universe_day != day:
            self._refresh_universe(tickers)
            self.last_universe_day = day

        hour_key = now.strftime("%Y%m%dT%H")
        if now.minute >= self.cfg.execution.rebalance_minute and self.last_cycle_hour != hour_key:
            self.last_cycle_hour = hour_key
            self.hourly_cycle()

        if now.hour >= NO_FILL_WARN_HOUR and self.last_warn_day != day:
            self.last_warn_day = day
            if self.store.kv_get("last_fill_day") != day:
                log.warning("no fill yet this UTC day (activity rule needs 1 trade/day on 8+ days)")

    def run_forever(self) -> None:
        self.startup()
        while True:
            started = time.monotonic()
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 -- the bot must keep running; full traceback is logged
                log.exception("tick failed", extra={"err": str(e)})
            time.sleep(max(1.0, LOOP_S - (time.monotonic() - started)))


def now_utc() -> datetime:
    return datetime.now(UTC)
