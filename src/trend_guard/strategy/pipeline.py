"""One hourly decision, end to end: signals -> regime -> gate -> portfolio -> overlays -> final weights.

This is the single pure function both engines call (BUILD_SPEC.md Section 0 rule 6): the backtest
engine feeds it precomputed signal/vol/covariance values bar by bar, the live engine feeds it the same
quantities computed from the latest closed bars. No I/O, no clock reads -- `now` arrives in
OverlayInputs.

Order of operations follows BUILD_SPEC Sections 6.3-7:
  regime(mkt, breadth) -> s_eff per asset (crypto gate or diversifier gate)
  -> build_portfolio (inverse vol, caps, shrunk-cov vol target, caps, gross)
  -> w * dd * day * event * endgame * tournament * crowd_i * news_i * risk_scale
  -> "no new risk" restrictions (hard daily brake, shock freezes, crowding short block, cooldowns)
  -> caps and gross again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from trend_guard.config import EventItem, StrategyConfig, SymbolsConfig, UniverseConfig
from trend_guard.risk.overlays import (
    crowding_long,
    crowding_short,
    daily_brake,
    drawdown_governor,
    endgame_lockin,
    event_overlay,
    tournament_overlay,
)
from trend_guard.risk.protective import atr
from trend_guard.strategy.portfolio import apply_aggregate_caps, apply_caps, build_portfolio, enforce_gross
from trend_guard.strategy.regime import breadth, gate_crypto, gate_diversifier, next_regime
from trend_guard.strategy.signals import ANN_H, compute_signal

CRYPTO, CRYPTO_BETA, BSTOCK, PAXG = "crypto", "crypto_beta", "bstock", "paxg"
_BTC_ETH = {"BTC/USD", "ETH/USD"}


# --- asset classification ------------------------------------------------------------------------
def classify(pair: str, symbols: SymbolsConfig) -> str:
    """Asset class used for gating: crypto-beta stocks use the crypto regime gate (Section 6.3)."""
    if symbols.asset_class.get(pair) == "gold":
        return PAXG
    if pair in symbols.crypto_beta_stocks:
        return CRYPTO_BETA
    if pair in symbols.bstocks:
        return BSTOCK
    return CRYPTO


def bucket_for(pair: str, asset_class: str, universe: UniverseConfig) -> str:
    """Cap bucket per Section 6.5. Crypto-beta stocks are equities, so they sit in the bStock bucket."""
    if asset_class == PAXG:
        return "paxg"
    if asset_class in (BSTOCK, CRYPTO_BETA):
        return "bstock"
    if pair in _BTC_ETH:
        return "btc_eth"
    if pair in universe.majors:
        return "major"
    return "other"


# --- covariance and channel helpers (shared by both engines) -------------------------------------
def ewma_alpha(halflife_h: float) -> float:
    """Same smoothing constant pandas uses for ewm(halflife=..., adjust=False)."""
    return 1.0 - math.exp(-math.log(2.0) / halflife_h)


def ewma_cov_update(prev: np.ndarray | None, r: np.ndarray, alpha: float) -> np.ndarray:
    """One step of the zero-mean EWMA covariance of hourly log returns (matches ewma_vol's
    zero-mean convention). Missing returns count as 0 so a halted asset doesn't poison the matrix."""
    r = np.nan_to_num(r, nan=0.0)
    outer = np.outer(r, r)
    return outer if prev is None else (1.0 - alpha) * prev + alpha * outer


def ewma_cov_annualised(log_returns: pd.DataFrame, halflife_h: float) -> pd.DataFrame:
    """Annualised EWMA covariance at the last row of `log_returns` (columns = assets)."""
    alpha = ewma_alpha(halflife_h)
    cov: np.ndarray | None = None
    for row in log_returns.to_numpy(dtype=float):
        cov = ewma_cov_update(cov, row, alpha)
    if cov is None:
        cov = np.zeros((len(log_returns.columns), len(log_returns.columns)))
    return pd.DataFrame(cov * ANN_H, index=log_returns.columns, columns=log_returns.columns)


def median_lookback(lookbacks_h: list[int]) -> int:
    """The trailing stop's Donchian reference uses the median lookback (Section 7.1)."""
    s = sorted(lookbacks_h)
    return s[len(s) // 2]


def donchian_mid(close: pd.Series, lookback_h: int) -> pd.Series:
    upper = close.rolling(lookback_h, min_periods=lookback_h).max()
    lower = close.rolling(lookback_h, min_periods=lookback_h).min()
    return (upper + lower) / 2.0


# --- signal by mode (shared by both engines) --------------------------------------------------------
def asset_signal(close: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """Hourly-indexed signal for one asset under cfg.signal.mode. Causal in every mode.

    Daily mode resamples hourly closes (indexed by bar open time) to daily closes, computes the
    ensemble over lookbacks_d, then shifts it one day forward so a daily value only becomes usable
    once that whole day's bars have closed. Vol half-life is converted from hours to days.
    """
    mode = cfg.signal.mode
    hourly = compute_signal(close, cfg.signal, cfg.vol) if mode in ("hourly", "blend") else None
    if mode == "hourly":
        return hourly
    daily_close = close.resample("1D").last().dropna()
    daily_sig_cfg = cfg.signal.model_copy(update={"lookbacks_h": cfg.signal.lookbacks_d})
    daily_vol_cfg = cfg.vol.model_copy(update={"halflife_h": max(1, round(cfg.vol.halflife_h / 24))})
    daily = compute_signal(daily_close, daily_sig_cfg, daily_vol_cfg)
    daily.index = daily.index + pd.Timedelta(days=1)  # usable only after the day has fully closed
    daily_h = daily.reindex(close.index.union(daily.index)).ffill().reindex(close.index)
    if mode == "daily":
        return daily_h
    return (hourly + daily_h) / 2.0  # blend: NaN until both are warm


def warmup_bars(cfg: StrategyConfig) -> int:
    """Hourly bars needed before the signal is defined under the configured mode (+50 of slack)."""
    need_h = max(cfg.signal.lookbacks_h) + 1
    need_d = (max(cfg.signal.lookbacks_d) + 2) * 24
    mode = cfg.signal.mode
    return (need_h if mode == "hourly" else need_d if mode == "daily" else max(need_h, need_d)) + 50


def atr_series(bars: pd.DataFrame, bar_h: int, window: int) -> pd.Series:
    """ATR on `bar_h`-hour bars, returned on the hourly index and causal: a coarse bar's range only
    counts once that bar has fully closed."""
    if bar_h <= 1:
        return atr(bars["high"], bars["low"], bars["close"], window)
    rule = f"{bar_h}h"
    agg = bars[["high", "low", "close"]].resample(rule).agg({"high": "max", "low": "min", "close": "last"})
    coarse = atr(agg["high"], agg["low"], agg["close"], window)
    coarse.index = coarse.index + pd.Timedelta(hours=bar_h)  # usable after the coarse bar closes
    return coarse.reindex(bars.index.union(coarse.index)).ffill().reindex(bars.index)


# --- trailing stop state (pure ratchet) ------------------------------------------------------------
@dataclass
class StopState:
    side: str  # "LONG" | "SHORT"
    best_price: float  # highest high since entry (long) / lowest low (short)
    stop_ref: float | None = None  # ratcheted Donchian mid; None until price has been on the right side


def ratchet(state: StopState, high: float, low: float, close: float, mid_now: float | None) -> StopState:
    """Advance a stop by one bar/tick, only ever in the trade's favour.

    The Donchian reference arms only once price closes beyond the channel mid, so a trade entered on
    a moderate signal while still under the mid is not stopped out on the very next bar.
    """
    if state.side == "LONG":
        best = max(state.best_price, high)
        ref = state.stop_ref
        if mid_now is not None and not math.isnan(mid_now) and mid_now < close:
            ref = mid_now if ref is None else max(ref, mid_now)
    else:
        best = min(state.best_price, low)
        ref = state.stop_ref
        if mid_now is not None and not math.isnan(mid_now) and mid_now > close:
            ref = mid_now if ref is None else min(ref, mid_now)
    return StopState(state.side, best, ref)


def stop_level(state: StopState, atr_value: float, atr_mult: float) -> float:
    """Price at which the protective loop exits (Section 7.1)."""
    if state.side == "LONG":
        atr_stop = state.best_price - atr_mult * atr_value
        return max(atr_stop, state.stop_ref) if state.stop_ref is not None else atr_stop
    atr_stop = state.best_price + atr_mult * atr_value
    return min(atr_stop, state.stop_ref) if state.stop_ref is not None else atr_stop


# --- the decision ---------------------------------------------------------------------------------
@dataclass
class OverlayInputs:
    now: datetime
    equity: float
    peak_equity: float
    pnl_today_pct: float  # P&L since 00:00 UTC as a fraction of that anchor
    competition_day: int  # 1-based
    cumulative_return: float  # since competition start
    events: list[EventItem] = field(default_factory=list)
    funding_z: dict[str, float | None] = field(default_factory=dict)
    oi_chg_24h: dict[str, float | None] = field(default_factory=dict)
    ret_72h: dict[str, float | None] = field(default_factory=dict)
    news_mult: dict[str, float] = field(default_factory=dict)  # per asset, from the optional LLM veto
    news_block: set[str] = field(default_factory=set)  # coins the veto forced to 0
    freeze_longs: bool = False  # shock filter: no new crypto longs
    freeze_shorts: bool = False  # squeeze filter: no new shorts
    cooldown_long: set[str] = field(default_factory=set)  # stopped out recently, no long re-entry
    cooldown_short: set[str] = field(default_factory=set)
    twentieth_place_return: float | None = None  # tournament overlay input (disabled by default)


@dataclass
class Decision:
    regime: str
    mkt: float
    breadth: float
    signals: dict[str, float]
    s_eff: dict[str, float]
    weights_raw: dict[str, float]
    weights_final: dict[str, float]
    multipliers: dict[str, tuple[float, str]]  # global overlays: name -> (mult, reason)
    per_asset: dict[str, dict]  # crowding / news multipliers and restrictions per asset
    no_new_risk: bool

    def to_json(self) -> dict:
        def clean(d: dict) -> dict:
            return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}

        return {
            "regime": self.regime,
            "mkt": None if math.isnan(self.mkt) else self.mkt,
            "breadth": self.breadth,
            "signals": clean(self.signals),
            "s_eff": clean(self.s_eff),
            "weights_raw": clean(self.weights_raw),
            "weights_final": clean(self.weights_final),
            "multipliers": {k: [m, r] for k, (m, r) in self.multipliers.items()},
            "per_asset": self.per_asset,
            "no_new_risk": self.no_new_risk,
        }


def _no_increase(w_new: float, w_cur: float, block_long: bool, block_short: bool) -> float:
    """Clamp a target so blocked directions can shrink or hold, never grow or open."""
    if block_long and w_new > 0:
        w_new = min(w_new, max(w_cur, 0.0))
    if block_short and w_new < 0:
        w_new = max(w_new, min(w_cur, 0.0))
    return w_new


def decide(
    signals: dict[str, float],
    mkt: float,
    prev_regime: str,
    sigma_ann: dict[str, float],
    cov: pd.DataFrame,
    asset_class: dict[str, str],
    bucket_of: dict[str, str],
    current_w: dict[str, float],
    shorts_enabled: bool,
    ov: OverlayInputs,
    cfg: StrategyConfig,
) -> Decision:
    """Compute final target weights for one hourly cycle. See module docstring for the order."""
    crypto_signals = {a: s for a, s in signals.items() if asset_class.get(a) == CRYPTO}
    br = breadth(crypto_signals)
    regime = prev_regime if math.isnan(mkt) else next_regime(prev_regime, mkt, br, cfg.regime)

    # 1. gate each asset by its class
    s_eff: dict[str, float] = {}
    for a, s in signals.items():
        if asset_class.get(a) in (CRYPTO, CRYPTO_BETA):
            s_eff[a] = gate_crypto(s, regime, cfg.regime, shorts_enabled)
            held = current_w.get(a, 0.0)
            if cfg.regime.hold_to_zero and s_eff[a] == 0.0 and s == s and held * s > 0:
                # exit band: keep a held position while its signal is still on the same side of zero
                b = cfg.regime.budgets[regime]
                budget = b.long if s > 0 else (b.short if shorts_enabled else 0.0)
                s_eff[a] = s * budget
        else:
            s_eff[a] = gate_diversifier(s, cfg.diversifiers)
            if cfg.regime.hold_to_zero and s_eff[a] == 0.0 and s == s and current_w.get(a, 0.0) * s > 0:
                s_eff[a] = s

    # 2. portfolio construction; NaN (no history) means flat
    raw = build_portfolio(s_eff, sigma_ann, cov, bucket_of, cfg.vol, cfg.portfolio)
    raw = {a: (0.0 if w != w else w) for a, w in raw.items()}

    # 3. global overlays
    risk = cfg.risk
    day = daily_brake(ov.pnl_today_pct, risk)
    no_new_risk = day[0] == 0.0  # hard brake: reductions only until the next UTC day
    mults: dict[str, tuple[float, str]] = {
        "dd_governor": drawdown_governor(ov.equity, ov.peak_equity, risk),
        "daily_brake": (1.0, day[1]) if no_new_risk else day,
        "event": event_overlay(ov.now, ov.events, risk),
        "endgame": endgame_lockin(ov.competition_day, ov.cumulative_return, risk),
        "risk_scale": (cfg.competition.risk_scale, f"risk_scale={cfg.competition.risk_scale}"),
    }
    if risk.tournament.enabled:
        mults["tournament"] = tournament_overlay(ov.cumulative_return, ov.twentieth_place_return, risk)
    global_mult = math.prod(m for m, _ in mults.values())

    # 4. per-asset overlays and restrictions
    final: dict[str, float] = {}
    per_asset: dict[str, dict] = {}
    for a, w in raw.items():
        info: dict = {}
        mult = global_mult
        block_short = False
        if w > 0:
            cm, cr = crowding_long(ov.funding_z.get(a), ov.oi_chg_24h.get(a), signals.get(a), risk)
        else:
            cm, cr, block_short = crowding_short(ov.funding_z.get(a), ov.ret_72h.get(a), risk)
        if cm != 1.0:
            info["crowding"] = [cm, cr]
        mult *= cm
        nm = ov.news_mult.get(a, 1.0)
        if nm != 1.0:
            info["news"] = nm
        mult *= nm
        w_new = w * mult

        is_crypto = asset_class.get(a) in (CRYPTO, CRYPTO_BETA)
        block_long = no_new_risk or a in ov.cooldown_long or (ov.freeze_longs and is_crypto)
        block_short = block_short or no_new_risk or a in ov.cooldown_short or ov.freeze_shorts
        w_new = _no_increase(w_new, current_w.get(a, 0.0), block_long, block_short)
        if a in ov.news_block:
            w_new = 0.0
            info["news_block"] = True
        if block_long or block_short:
            info["blocked"] = {"long": block_long, "short": block_short}
        if info:
            per_asset[a] = info
        final[a] = w_new

    # 5. overlays only scale down, but restrictions can reshuffle -- re-enforce hard limits
    final = apply_caps(final, bucket_of, cfg.portfolio)
    final = apply_aggregate_caps(final, bucket_of, cfg.portfolio)
    final = enforce_gross(final, cfg.portfolio.gross_max)

    return Decision(regime, mkt, br, signals, s_eff, raw, final, mults, per_asset, no_new_risk)
