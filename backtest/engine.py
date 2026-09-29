"""Hourly event-driven backtest. BUILD_SPEC.md Section 11.2.

Uses the exact strategy/risk/order-planning code the live bot uses:
  strategy.pipeline.asset_signal / decide  (signals, regime, gates, portfolio, overlays)
  execution.order_manager.plan_orders      (deadband, min-hold, flips, no-leverage budget)
  execution.sim_broker.SimBroker           (Section 11.2 fill model)

Timeline for each hourly bar t (index = bar open time; its close is the decision point):
  1. advance_bar(t) resolves orders submitted at t-1: maker limits fill at their price if bar t
     trades through them, otherwise at bar t close + slippage as taker (the reprice->MARKET path).
  2. protective stops are checked against bar t's high/low using the stop level set BEFORE bar t
     (conservative: a touched stop fills at the stop, or at the open if the bar gapped through it).
  3. equity is marked at bar t close.
  4. the decision is made on data through bar t and orders are submitted for bar t+1.

Simulation choices that differ from live (all documented in the README's caveats):
  - Endgame, drawdown peak and "competition day" run on consecutive 14-day blocks from the start,
    so each block looks like a fresh competition to the overlays.
  - The shock filter uses the hourly BTC return (z = r_1h / sigma_h); live uses 15-minute ticks.
  - The spread filter in the universe is skipped (no historical Roostoo spreads); membership is the
    top `max_crypto` by trailing 30-day Binance quote volume, recomputed daily.
  - Crowding needs funding + open-interest history; without it the overlay is neutral (1.0).
  - Shorts open/close at bar t close with slippage (market-only on the exchange too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from trend_guard.clock import SimClock
from trend_guard.config import StrategyConfig, SymbolsConfig, load_events
from trend_guard.execution.order_manager import plan_orders
from trend_guard.execution.sim_broker import OHLCBar, SimBroker
from trend_guard.risk.protective import atr
from trend_guard.strategy.pipeline import (
    CRYPTO,
    PAXG,
    OverlayInputs,
    StopState,
    asset_signal,
    bucket_for,
    classify,
    decide,
    donchian_mid,
    ewma_alpha,
    ewma_cov_update,
    median_lookback,
    ratchet,
    stop_level,
)
from trend_guard.strategy.regime import NEUTRAL
from trend_guard.strategy.signals import ANN_H, ewma_vol

STARTING_CASH = Decimal("100000")
QV_WINDOW_H = 30 * 24  # trailing quote-volume window for liquidity ranking (universe + index)
ATR_WINDOW = 24  # Section 7.1: ATR over the last 24 hourly bars
BLOCK_DAYS = 14  # competition length, used to reset endgame/peak per block


@dataclass
class Variant:
    """Ablations required by Section 11.3's baselines."""

    name: str = "strategy"
    long_only: bool = False
    overlays: bool = True  # False = no drawdown/brake/event/endgame/crowding and no protective stops


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series  # hourly equity at bar close, UTC
    trades: pd.DataFrame  # one row per fill
    regimes: pd.Series  # regime per hour
    exposure: pd.DataFrame  # gross long / short collateral as fraction of equity, hourly
    fees: float = 0.0
    meta: dict = field(default_factory=dict)


def load_bars(
    bars_dir: Path, symbols: SymbolsConfig, pairs: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Load data/bars/1h/{SYM}.parquet into {roostoo_pair: OHLCV frame}. Pair = COIN/USD by default."""
    reverse = {v: k for k, v in symbols.overrides.items()}
    out: dict[str, pd.DataFrame] = {}
    for f in sorted(Path(bars_dir).glob("*.parquet")):
        sym = f.stem
        pair = reverse.get(sym) or (sym[:-4] + "/USD" if sym.endswith("USDT") else None)
        if pair is None or pair in symbols.stablecoins or (pairs and pair not in pairs):
            continue
        df = pd.read_parquet(f)
        df.index = (
            pd.DatetimeIndex(df.index).tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        )
        out[pair] = df[["open", "high", "low", "close", "quote_volume"]].astype(float).sort_index()
    return out


def _panel(bars: dict[str, pd.DataFrame], col: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    # forward-fill at most 2 bars (Section 5.3 continuity rule); longer gaps stay NaN = untradeable
    return pd.DataFrame({p: df[col].reindex(index) for p, df in bars.items()}).ffill(limit=2)


def run_backtest(
    bars: dict[str, pd.DataFrame],
    cfg: StrategyConfig,
    symbols: SymbolsConfig,
    start: str,
    end: str | None = None,
    variant: Variant | None = None,
) -> BacktestResult:
    variant = variant or Variant()
    pairs = sorted(bars)
    first = min(df.index[0] for df in bars.values())
    last = max(df.index[-1] for df in bars.values())
    index = pd.date_range(first, last, freq="1h")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else last

    # --- precompute everything that is a causal function of past data -------------------------
    close = _panel(bars, "close", index)
    openp = _panel(bars, "open", index)
    high = _panel(bars, "high", index)
    low = _panel(bars, "low", index)
    qv = pd.DataFrame({p: df["quote_volume"].reindex(index) for p, df in bars.items()}).fillna(0.0)
    log_ret = np.log(close / close.shift(1))

    asset_class = {p: classify(p, symbols) for p in pairs}
    bucket_of = {p: bucket_for(p, asset_class[p], cfg.universe) for p in pairs}
    signals = pd.DataFrame({p: asset_signal(close[p].dropna(), cfg).reindex(index) for p in pairs})
    sigma_h = pd.DataFrame({p: ewma_vol(log_ret[p], cfg.vol.halflife_h) for p in pairs})
    sigma_ann = sigma_h * math.sqrt(ANN_H)
    atr_df = pd.DataFrame({p: atr(high[p], low[p], close[p], ATR_WINDOW) for p in pairs})
    mid_df = pd.DataFrame({p: donchian_mid(close[p], median_lookback(cfg.signal.lookbacks_h)) for p in pairs})
    ret_72h = close / close.shift(72) - 1.0
    first_seen = close.apply(lambda s: s.first_valid_index())

    # liquidity ranking (crypto only), lagged one bar so it never uses the current bar's volume
    crypto_pairs = [p for p in pairs if asset_class[p] == CRYPTO]
    qv_roll = qv[crypto_pairs].rolling(QV_WINDOW_H, min_periods=24).sum().shift(1)
    rank = qv_roll.rank(axis=1, ascending=False)

    # market index: equal-weight log return of the top index_size crypto, signal via the same code
    in_index = rank <= cfg.regime.index_size
    idx_ret = (log_ret[crypto_pairs].where(in_index)).mean(axis=1).fillna(0.0)
    idx_price = np.exp(idx_ret.cumsum())
    mkt_sig = asset_signal(idx_price, cfg)

    btc = "BTC/USD" if "BTC/USD" in pairs else None
    events = load_events()

    # --- simulation state -----------------------------------------------------------------------
    run_index = index[(index >= start_ts) & (index <= end_ts)]
    clock = SimClock(run_index[0].to_pydatetime())
    broker = SimBroker(STARTING_CASH, cfg.backtest, clock, mode="backtest")
    alpha = ewma_alpha(cfg.portfolio.cov_halflife_h)
    cov: np.ndarray | None = None
    # warm the covariance on the pre-start history so the first decision isn't built on one outer product
    for row in log_ret.loc[index < start_ts, pairs].to_numpy()[-2000:]:
        cov = ewma_cov_update(cov, row, alpha)

    regime = NEUTRAL
    eligible: set[str] = set()
    last_trade: dict = {}
    stops: dict[str, StopState] = {}
    cooldown_long: dict[str, pd.Timestamp] = {}
    cooldown_short: dict[str, pd.Timestamp] = {}
    freeze_longs_until = freeze_shorts_until = run_index[0] - timedelta(hours=1)
    block_start_equity = float(STARTING_CASH)
    block_peak = float(STARTING_CASH)
    day_anchor = float(STARTING_CASH)
    min_hist = timedelta(days=cfg.universe.min_history_days)

    eq_rec: list[float] = []
    reg_rec: list[str] = []
    expo_rec: list[tuple[float, float]] = []
    trades: list[dict] = []
    fees = 0.0
    min_cash = float(STARTING_CASH)
    half = cfg.backtest.half_spread

    def log_fill(ts, pair, action, qty, price, fee, role, reason):
        nonlocal fees
        fees += float(fee)
        trades.append(
            {
                "ts": ts,
                "pair": pair,
                "action": action,
                "qty": float(qty),
                "price": float(price),
                "notional": float(qty) * float(price),
                "fee": float(fee),
                "role": role,
                "reason": reason,
            }
        )

    for i, t in enumerate(run_index):
        clock.set(t.to_pydatetime())
        prev_t = run_index[i - 1] if i else None

        # 1. resolve orders queued at t-1 against bar t
        for p in pairs:
            c = close.at[t, p]
            if c != c:
                continue
            bar = OHLCBar(
                ts=t.to_pydatetime(),
                open=Decimal(str(openp.at[t, p])),
                high=Decimal(str(high.at[t, p])),
                low=Decimal(str(low.at[t, p])),
                close=Decimal(str(c)),
            )
            for ack in broker.advance_bar(p, bar):
                o = broker.query(ack.order_id)
                log_fill(t, p, o.side, ack.filled_qty, ack.avg_price, ack.fee, ack.role, "rebalance")

        exp = broker.get_exposure()

        # 2. protective stops on bar t's range, using levels set before bar t
        if variant.overlays:
            for p in list(stops):
                st = stops[p]
                holding_long = exp.long_qty.get(p, Decimal(0)) > 0
                holding_short = exp.short_collateral.get(p, Decimal(0)) > 0
                if (st.side == "LONG" and not holding_long) or (st.side == "SHORT" and not holding_short):
                    stops.pop(p)
                    continue
                c = close.at[t, p]
                a = atr_df.at[prev_t, p] if prev_t is not None else float("nan")
                if c != c or a != a:
                    continue
                level = stop_level(st, a, cfg.protective.atr_mult)
                o_, h_, l_ = openp.at[t, p], high.at[t, p], low.at[t, p]
                if st.side == "LONG" and l_ <= level:
                    px = min(level, o_)
                    ack = broker.fill_now(p, "SELL", exp.long_qty[p], Decimal(str(px)))
                    log_fill(t, p, "SELL", ack.filled_qty, ack.avg_price, ack.fee, "TAKER", "trailing_stop")
                    cooldown_long[p] = t + timedelta(hours=cfg.protective.cooldown_h)
                    last_trade[p] = t.to_pydatetime()
                    stops.pop(p)
                elif st.side == "SHORT" and h_ >= level:
                    broker.last_price[p] = Decimal(str(max(level, o_)))  # close at the stop, not the close
                    ack = broker.short_close(p, Decimal(100))
                    log_fill(
                        t,
                        p,
                        "SHORT_CLOSE",
                        ack.closed_qty,
                        ack.close_price,
                        ack.close_fee,
                        "TAKER",
                        "trailing_stop",
                    )
                    broker.last_price[p] = Decimal(str(c))
                    cooldown_short[p] = t + timedelta(hours=cfg.protective.cooldown_h)
                    last_trade[p] = t.to_pydatetime()
                    stops.pop(p)
                else:
                    stops[p] = ratchet(st, h_, l_, c, mid_df.at[t, p])

            # shock / squeeze filter on the hourly BTC move
            if btc and prev_t is not None:
                r1 = log_ret.at[t, btc]
                sh = sigma_h.at[prev_t, btc]
                if r1 == r1 and sh == sh and sh > 0:
                    z = r1 / sh
                    exp = broker.get_exposure()
                    if z < -cfg.protective.shock_z:
                        for p, q in exp.long_qty.items():
                            if asset_class.get(p) == CRYPTO and q > 0:
                                ack = broker.fill_now(p, "SELL", q / 2, Decimal(str(close.at[t, p])))
                                log_fill(
                                    t, p, "SELL", ack.filled_qty, ack.avg_price, ack.fee, "TAKER", "shock"
                                )
                        freeze_longs_until = t + timedelta(minutes=60)
                    r72 = ret_72h.at[t, btc]
                    if z > cfg.protective.shock_z and r72 == r72 and r72 < cfg.protective.squeeze_prior_drop:
                        for p in list(exp.short_collateral):
                            ack = broker.short_close(p, Decimal(50))
                            log_fill(
                                t,
                                p,
                                "SHORT_CLOSE",
                                ack.closed_qty,
                                ack.close_price,
                                ack.close_fee,
                                "TAKER",
                                "squeeze",
                            )
                        freeze_shorts_until = t + timedelta(minutes=60)

        # 3. mark equity
        exp = broker.get_exposure()
        equity = float(exp.equity)
        if t.hour == 0:
            day_anchor = equity
        block_idx = (t - run_index[0]) // timedelta(days=BLOCK_DAYS)
        block_first = run_index[0] + block_idx * timedelta(days=BLOCK_DAYS)
        if t == block_first:
            block_start_equity = equity
            block_peak = equity
        block_peak = max(block_peak, equity)
        eq_rec.append(equity)
        longs = float(sum((exp.long_notional(p) for p in exp.long_qty), Decimal(0)))
        shorts = float(sum(exp.short_collateral.values(), Decimal(0)))
        expo_rec.append((longs / equity if equity else 0.0, shorts / equity if equity else 0.0))
        # no leverage means we never spend cash we don't have. Collateral isn't marked to market, so
        # longs + collateral can exceed equity while a short is losing without any borrowing; the
        # real invariant is free cash >= 0. A taker fallback fill one bar later can overshoot the
        # planned budget slightly -- tracked and reported rather than hidden.
        min_cash = min(min_cash, float(exp.cash))

        # 4. universe refresh daily, decision every hour
        if t.hour == 0 or not eligible:
            r = rank.loc[t]
            keep = set()
            for p in pairs:
                fs = first_seen[p]
                if fs is None or t - fs < min_hist or close.at[t, p] != close.at[t, p]:
                    continue
                if asset_class[p] == PAXG:
                    keep.add(p)
                elif asset_class[p] == CRYPTO:
                    rk = r.get(p, np.nan)
                    limit = cfg.universe.max_crypto + (5 if p in eligible else 0)
                    if rk == rk and rk <= limit:
                        keep.add(p)
                elif cfg.universe.bstocks_enabled:
                    keep.add(p)
            eligible = keep

        cov = ewma_cov_update(cov, log_ret.loc[t, pairs].to_numpy(dtype=float), alpha)
        if t == run_index[-1]:
            reg_rec.append(regime)
            break
        el = sorted(eligible)
        sig_now = {p: float(signals.at[t, p]) for p in el}
        sa_now = {p: float(sigma_ann.at[t, p]) for p in el}
        pos = [pairs.index(p) for p in el]
        cov_df = pd.DataFrame(cov[np.ix_(pos, pos)] * ANN_H, index=el, columns=el)
        current_w = {p: float(exp.long_notional(p)) / equity for p in exp.long_qty}
        for p, c_ in exp.short_collateral.items():
            current_w[p] = current_w.get(p, 0.0) - float(c_) / equity

        tpy = t.to_pydatetime()
        ov = OverlayInputs(
            now=tpy,
            equity=equity if variant.overlays else block_peak,
            peak_equity=block_peak,
            pnl_today_pct=(equity / day_anchor - 1.0) if variant.overlays else 0.0,
            competition_day=int((t - block_first) / timedelta(days=1)) + 1 if variant.overlays else 1,
            cumulative_return=equity / block_start_equity - 1.0,
            events=events if variant.overlays else [],
            ret_72h={p: float(ret_72h.at[t, p]) for p in el},
            freeze_longs=t < freeze_longs_until,
            freeze_shorts=t < freeze_shorts_until,
            cooldown_long={p for p, u in cooldown_long.items() if t < u},
            cooldown_short={p for p, u in cooldown_short.items() if t < u},
        )
        mkt_now = float(mkt_sig.at[t]) if not math.isnan(mkt_sig.at[t]) else float("nan")
        dec = decide(
            sig_now,
            mkt_now,
            regime,
            sa_now,
            cov_df,
            asset_class,
            bucket_of,
            current_w,
            not variant.long_only,
            ov,
            cfg,
        )
        regime = dec.regime
        reg_rec.append(regime)

        plans = plan_orders(dec.weights_final, exp, cfg.execution, tpy, last_trade)
        for pl in plans:
            px = close.at[t, pl.pair]
            if px != px:
                continue
            if pl.action == "BUY":
                broker.submit_limit(pl.pair, "BUY", pl.qty, Decimal(str(px * (1 - half))))
            elif pl.action == "SELL":
                broker.submit_limit(pl.pair, "SELL", pl.qty, Decimal(str(px * (1 + half))))
            elif pl.action == "SHORT_OPEN":
                ack = broker.short_open(pl.pair, pl.collateral)
                log_fill(
                    t,
                    pl.pair,
                    "SHORT_OPEN",
                    pl.collateral / ack.entry_price,
                    ack.entry_price,
                    ack.open_fee,
                    "TAKER",
                    pl.reason,
                )
                if variant.overlays and pl.pair not in stops:
                    stops[pl.pair] = StopState("SHORT", float(ack.entry_price))
            else:
                ack = broker.short_close(pl.pair, pl.close_pct)
                log_fill(
                    t,
                    pl.pair,
                    "SHORT_CLOSE",
                    ack.closed_qty,
                    ack.close_price,
                    ack.close_fee,
                    "TAKER",
                    pl.reason,
                )
            last_trade[pl.pair] = tpy
            if pl.action == "BUY" and variant.overlays and pl.pair not in stops:
                stops[pl.pair] = StopState("LONG", float(px))

    eq = pd.Series(eq_rec, index=run_index[: len(eq_rec)], name="equity")
    expo = pd.DataFrame(expo_rec, index=eq.index, columns=["long", "short"])
    trades_df = pd.DataFrame(trades)
    meta = {"min_cash": min_cash, "n_trades": len(trades_df)}
    return BacktestResult(variant.name, eq, trades_df, pd.Series(reg_rec, index=eq.index), expo, fees, meta)


# --- baselines ------------------------------------------------------------------------------------
def buy_and_hold(bars: dict[str, pd.DataFrame], pair: str, start: str, end: str | None = None) -> pd.Series:
    c = bars[pair]["close"]
    c = c[c.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        c = c[c.index <= pd.Timestamp(end, tz="UTC")]
    taker_fee = 0.001  # one market buy on day one
    return float(STARTING_CASH) * (1 - taker_fee) * c / c.iloc[0]


def equal_weight_hold(
    bars: dict[str, pd.DataFrame], symbols: SymbolsConfig, start: str, n: int = 10, end: str | None = None
) -> pd.Series:
    """Equal-weight buy-and-hold of the n most liquid crypto (by 30-day quote volume before start)."""
    start_ts = pd.Timestamp(start, tz="UTC")
    liq = {}
    for p, df in bars.items():
        if classify(p, symbols) != CRYPTO:
            continue
        pre = df[(df.index < start_ts) & (df.index >= start_ts - timedelta(days=30))]
        if len(pre) > 24 * 20:
            liq[p] = pre["quote_volume"].sum()
    top = sorted(liq, key=liq.get, reverse=True)[:n]
    curves = [buy_and_hold(bars, p, start, end) / len(top) for p in top]
    return pd.concat(curves, axis=1).ffill().sum(axis=1)


def cash(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(float(STARTING_CASH), index=index)
