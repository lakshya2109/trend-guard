"""Live/paper orchestration: a 60-second loop plus the hourly cycle. BUILD_SPEC.md Section 10.3.

The engine owns all I/O (Roostoo, Binance, SQLite, logs, the clock). Every trading decision comes
from pure code shared with the backtest: strategy.book.book_weights (targets) and
execution.order_manager.plan_orders (the contest trade rule), so what is tested is what trades.

Loop (every 60 s):
  HALT file present -> cancel our resting orders and do nothing else
  refresh tickers -> equity snapshot -> service resting limit orders (reprice / escalate)
  every 5 min: clock sync
  at minute == rebalance_minute (once per hour): hourly cycle
     top up the Binance bar cache -> book targets -> x risk_scale x breaker -> plan -> execute
  activity rule: at minute ACTIVITY_MINUTE inside an activity window (00-03 / 12-15 UTC) with no fill
     since the window opened, one small MARKET order toward the book's own target (every day active)
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from trend_guard.clock import Clock
from trend_guard.config import EnvConfig, StrategyConfig, SymbolsConfig, config_hash
from trend_guard.data.binance_live import panel, update_cache
from trend_guard.execution.broker import Broker, Exposure
from trend_guard.execution.order_manager import (
    OrderExecutor,
    PlannedOrder,
    activity_due,
    activity_order,
    plan_orders,
)
from trend_guard.execution.sim_broker import SimBroker
from trend_guard.risk.breaker import Breaker
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.errors import ShortsDisabledError
from trend_guard.roostoo.models import Ticker
from trend_guard.roostoo.rules import TradingRule, parse_trading_rules
from trend_guard.state.store import Store
from trend_guard.strategy.book import bear_regime, book_weights
from trend_guard.universe import binance_symbol, tradable_pairs

log = logging.getLogger("engine")
dlog = logging.getLogger("decisions")

LOOP_S = 60
CLOCK_SYNC_EVERY = timedelta(minutes=5)
NO_FILL_WARN_HOUR = 18
ACTIVITY_MINUTE = 15  # after the cycle at :03 and its limit orders (repriced 4 x 60 s, then MARKET)
UNIVERSE_REFRESH_HOUR = 0  # re-read exchangeInfo once a day
FETCH_WORKERS = 8


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
        self.commit = git_commit()  # every fill row and decision carries this (Screen 1 traceability)
        self.rules: dict[str, TradingRule] = {}
        self.pairs: list[str] = []
        self.shorts_enabled = cfg.book.allow_short and store.kv_get("shorts_disabled") != "1"
        self.breaker = Breaker(cfg.breaker)
        self.breaker.load(json.loads(store.kv_get("breaker") or "{}"))
        self.last_cycle_hour: str | None = None
        self.last_clock_sync: datetime | None = None
        self.last_universe_day: str | None = None
        self.last_warn_day: str | None = None
        self.last_activity_hour: str | None = None
        self.executor = OrderExecutor(
            broker=broker,
            exec_cfg=cfg.execution,
            clock=clock,
            rules=self.rules,
            on_order=self._on_order,
            on_fill=self._on_fill,
        )

    # --- persistence callbacks -------------------------------------------------------------------
    def _on_order(self, record: dict) -> None:
        self.store.insert_order(record)

    def _on_fill(
        self, pair: str, side: str, qty: Decimal, price: Decimal, fee: Decimal, role: str, reason: str
    ) -> None:
        now = self.clock.now()
        self.store.kv_set("last_fill_day", now.strftime("%Y-%m-%d"))
        self.store.kv_set("last_fill_ts", now.isoformat())
        self.fills_dir.mkdir(parents=True, exist_ok=True)
        path = self.fills_dir / f"{now:%Y-%m-%d}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "pair", "side", "qty", "price", "fee", "role", "reason", "commit"])
            w.writerow([now.isoformat(), pair, side, qty, price, fee, role, reason, self.commit[:12]])

    # --- startup ------------------------------------------------------------------------------
    def startup(self) -> None:
        log.info(
            "startup",
            extra={
                "commit": self.commit,
                "config_hash": config_hash(self.config_path),
                "mode": self.env.bot_mode,
                "risk_scale": self.cfg.competition.risk_scale,
                "shorts_enabled": self.shorts_enabled,
            },
        )
        self.client.sync_clock()
        self.last_clock_sync = self.clock.now()
        self._refresh_universe()
        if isinstance(self.broker, SimBroker):
            self.broker.mark_to_market(self.client.get_ticker())
        exp = self.broker.get_exposure()
        if self.store.kv_get("start_equity") is None:
            self.store.kv_set("start_equity", exp.equity)
            self.store.kv_set("start_ts", self.clock.now().isoformat())

    def _refresh_universe(self) -> None:
        info = self.client.exchange_info()
        self.rules.update(parse_trading_rules(info))
        self.pairs = tradable_pairs(info, self.symbols)
        self.last_universe_day = self.clock.now().strftime("%Y-%m-%d")
        log.info("universe", extra={"pairs": len(self.pairs)})

    # --- data ---------------------------------------------------------------------------------
    def _load_panel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Top up the Binance cache for every tradable pair and return the recent close/volume panel."""
        now = self.clock.now()
        start = now - timedelta(hours=self.cfg.execution.history_hours)

        def one(pair: str) -> tuple[str, str, pd.DataFrame | None]:
            sym = binance_symbol(pair, self.symbols)
            try:
                return pair, sym, update_cache(sym, self.bars_dir, start, now)
            except RuntimeError as e:  # one missing feed must not stop the book
                log.warning("bars unavailable", extra={"pair": pair, "err": str(e)})
                return pair, sym, None

        bars, pair_of = {}, {}
        with ThreadPoolExecutor(FETCH_WORKERS) as pool:  # ~65 small requests: parallel keeps it to seconds
            for pair, sym, df in pool.map(one, self.pairs):
                if df is not None and len(df):
                    bars[sym] = df[df.index >= pd.Timestamp(start)]
                    pair_of[sym] = pair
        close, qv = panel(bars, pair_of)
        if len(close):
            # extend to the last closed hour so a feed that stopped updating shows as missing, not stale
            last_bar = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1)
            close = close.reindex(pd.date_range(close.index[0], last_bar, freq="1h", tz="UTC"))
            qv = qv.reindex(close.index)
        return close, qv

    # --- equity -------------------------------------------------------------------------------
    def _snapshot(self, exp: Exposure) -> None:
        now = self.clock.now()
        peak = max(Decimal(self.store.kv_get("peak") or exp.equity), exp.equity)
        self.store.kv_set("peak", peak)
        longs = sum((exp.long_notional(p) for p in exp.long_qty), Decimal(0))
        shorts = sum(exp.short_collateral.values(), Decimal(0))
        dd = 1 - exp.equity / peak if peak > 0 else Decimal(0)
        self.store.record_equity(
            now.replace(second=0, microsecond=0).isoformat(), exp.equity, exp.cash, longs, shorts, peak, dd
        )

    # --- hourly cycle --------------------------------------------------------------------------
    def hourly_cycle(self, force_act: bool = False) -> list[PlannedOrder]:
        now = self.clock.now()
        cycle_id = now.strftime("%Y%m%dT%H")
        close, qv = self._load_panel()
        if close.empty:
            log.error("no market data; skipping cycle")
            return []
        raw = book_weights(close, qv, self.cfg.book, self.shorts_enabled).iloc[-1]
        exp = self.broker.get_exposure()
        mult = self.breaker.update(float(exp.equity)) * self.cfg.competition.risk_scale
        self.store.kv_set("breaker", json.dumps(self.breaker.state()))
        targets = {p: float(w) * mult for p, w in raw.items() if abs(w) > 1e-12}
        prev = json.loads(self.store.kv_get("prev_targets") or "{}")
        # full rebalance when the bar that just closed is a decision bar (open time 00:00 / 12:00 UTC), the
        # same rule the backtest scores (backtest/contest.py acts on the bar's open hour)
        act_all = force_act or not prev or close.index[-1].hour in self.cfg.execution.decision_hours_utc
        # pairs the cash budget cut last hour count as changed, so they finish now (NaN never matches)
        retry = json.loads(self.store.kv_get("budget_retry") or "[]")
        prev_plan = {**prev, **{p: float("nan") for p in retry}}
        short_of: set[str] = set()
        plans = plan_orders(targets, exp, self.cfg.execution, act_all, prev_plan, self.rules, short_of)
        self.store.kv_set("prev_targets", json.dumps(targets))
        self.store.kv_set("budget_retry", json.dumps(sorted(short_of)))

        payload = {
            "cycle_id": cycle_id,
            "commit": self.commit[:12],
            "equity": float(exp.equity),
            "bar": str(close.index[-1]),
            "multiplier": mult,
            "breaker": self.breaker.state(),
            "act_all": act_all,
            "weights_final": targets,
            "gross": float(np.abs(raw).sum() * mult),
            "net": float(raw.sum() * mult),
            "bear_mode": bool(
                self.cfg.book.bear_mode.enabled and bear_regime(close, qv, self.cfg.book.bear_mode)[-1]
            ),
            "orders_planned": [
                {
                    "pair": o.pair,
                    "action": o.action,
                    "qty": str(o.qty),
                    "collateral": str(o.collateral),
                    "close_pct": str(o.close_pct),
                    "notional": str(o.notional),
                    "urgent": o.urgent,
                    "reason": o.reason,
                }
                for o in plans
            ],
        }
        self.store.record_decision(cycle_id, now.isoformat(), "book", payload)
        dlog.info("cycle", extra=json.loads(json.dumps(payload, default=str)))
        self._execute(plans, cycle_id)
        return plans

    def activity_check(self) -> PlannedOrder | None:
        """Keep every contest day active: inside an activity window with no fill since it opened, send
        one small MARKET order toward the book's own latest targets (see order_manager.activity_order)."""
        now = self.clock.now()
        raw_last = self.store.kv_get("last_fill_ts")
        last = datetime.fromisoformat(raw_last) if raw_last else None
        if not activity_due(now, last, self.cfg.execution.activity_windows_utc):
            return None
        stored = self.store.kv_get("prev_targets")
        if stored is None:  # no cycle has run yet: an empty target would read as "sell everything"
            log.warning("activity rule: no targets yet; waiting for the hourly cycle")
            return None
        targets = json.loads(stored)
        exp = self.broker.get_exposure()
        order = activity_order(targets, exp, self.cfg.execution, self.rules)
        cycle_id = now.strftime("%Y%m%dT%H") + "-activity"
        if order is None:
            log.error("activity rule: no order possible this hour", extra={"equity": str(exp.equity)})
            return None
        log.warning(
            "activity rule: no fill yet in this window, trading toward target",
            extra={"pair": order.pair, "action": order.action, "notional": str(order.notional)},
        )
        self.store.record_decision(
            cycle_id,
            now.isoformat(),
            "activity",
            {
                "cycle_id": cycle_id,
                "commit": self.commit[:12],
                "pair": order.pair,
                "action": order.action,
                "notional": str(order.notional),
                "reason": order.reason,
            },
        )
        self._execute([order], cycle_id)
        return order

    def _execute(self, plans: list[PlannedOrder], cycle_id: str) -> None:
        self.executor.shorts_enabled = self.shorts_enabled
        try:
            self.executor.execute(plans, cycle_id)
        except ShortsDisabledError as e:
            # Section 4.2: the exchange refuses shorts -> long-only book for the rest of the session
            self.shorts_enabled = False
            self.store.kv_set("shorts_disabled", "1")
            log.error("shorts disabled by exchange; short sleeve off for the session", extra={"err": str(e)})
            self.executor.execute([p for p in plans if p.action in ("BUY", "SELL")], cycle_id)

    # --- the loop -------------------------------------------------------------------------------
    def tick(self) -> None:
        now = self.clock.now()
        if self.halt_path.exists():
            self.executor.cancel_all_working()
            log.warning("HALT file present; idle", extra={"path": str(self.halt_path)})
            return

        tickers: dict[str, Ticker] = self.client.get_ticker()
        if isinstance(self.broker, SimBroker):
            self.broker.mark_to_market(tickers)
        self._snapshot(self.broker.get_exposure())
        self.executor.service()

        if self.last_clock_sync is None or now - self.last_clock_sync >= CLOCK_SYNC_EVERY:
            self.client.sync_clock()
            self.last_clock_sync = now

        day = now.strftime("%Y-%m-%d")
        if now.hour == UNIVERSE_REFRESH_HOUR and self.last_universe_day != day:
            self._refresh_universe()

        hour_key = now.strftime("%Y%m%dT%H")
        if now.minute >= self.cfg.execution.rebalance_minute and self.last_cycle_hour != hour_key:
            first = self.last_cycle_hour is None  # first cycle since start: full rebalance to the targets
            self.last_cycle_hour = hour_key
            self.hourly_cycle(force_act=first)

        if now.minute >= ACTIVITY_MINUTE and self.last_activity_hour != hour_key:
            self.last_activity_hour = hour_key
            self.activity_check()

        if now.hour >= NO_FILL_WARN_HOUR and self.last_warn_day != day:
            self.last_warn_day = day
            if self.store.kv_get("last_fill_day") != day:
                log.warning("no fill yet this UTC day (the contest needs trades on 8+ days)")

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
