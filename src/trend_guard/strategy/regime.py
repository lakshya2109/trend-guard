"""Market regime: synthetic index signal, breadth, hysteresis state machine, per-asset gates.

BUILD_SPEC.md Section 6.3. Pure functions only: the hysteresis state machine takes `prev_regime`
explicitly (no hidden state) so backtest (looping over bars, carrying its own `prev_regime` variable)
and live (engine loads `prev_regime` from the state store each cycle) call it identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import DiversifiersConfig, RegimeConfig, SignalConfig, VolConfig
from trend_guard.strategy.signals import compute_signal

RISK_ON = "RISK_ON"
RISK_OFF = "RISK_OFF"
NEUTRAL = "NEUTRAL"


def build_market_index(closes: dict[str, pd.Series]) -> pd.Series:
    """Synthetic equal-weight index price series from a set of eligible crypto close series.

    `closes` maps asset -> hourly close series (already restricted by the caller to the top
    `regime_cfg.index_size` eligible crypto, excluding PAXG and bStocks, per BUILD_SPEC 6.3).
    Equal-weight mean of hourly log returns, cumulated into a synthetic price starting at 1.0.
    Series are aligned on their union index; an asset missing a bar contributes no return that hour
    (treated as 0 log return for the mean, i.e. excluded from that hour's average) rather than
    dragging the index down with a spurious return computed across a data gap.
    """
    log_rets = []
    for close in closes.values():
        log_rets.append(np.log(close / close.shift(1)))
    ret_df = pd.concat(log_rets, axis=1)
    mkt_log_ret = ret_df.mean(axis=1, skipna=True)
    mkt_log_ret = mkt_log_ret.fillna(0.0)
    synthetic_price = np.exp(mkt_log_ret.cumsum())
    synthetic_price.iloc[0] = 1.0  # anchor start (first log ret is NaN->0, so cumsum starts flat)
    return synthetic_price


def market_signal(closes: dict[str, pd.Series], signal_cfg: SignalConfig, vol_cfg: VolConfig) -> pd.Series:
    """mkt: the ensemble trend signal applied to the synthetic market index. BUILD_SPEC 6.3."""
    index_price = build_market_index(closes)
    return compute_signal(index_price, signal_cfg, vol_cfg)


def breadth(signals: dict[str, float]) -> float:
    """Share of eligible crypto with a positive signal. NaN signals (insufficient history) don't count
    as eligible-and-positive, but they do count in the denominator since the caller already restricted
    `signals` to the eligible universe."""
    if not signals:
        return 0.0
    values = np.array(list(signals.values()), dtype=float)
    positive = np.sum(values > 0)  # NaN > 0 is False, so NaNs are naturally excluded from the count
    return float(positive) / float(len(values))


def next_regime(prev_regime: str, mkt: float, breadth_value: float, regime_cfg: RegimeConfig) -> str:
    """Hysteresis state machine, BUILD_SPEC Section 6.3. Explicit `prev_regime` in, new regime out."""
    on, off, hyst = regime_cfg.on, regime_cfg.off, regime_cfg.hysteresis

    if mkt > on.mkt and breadth_value > on.breadth:
        return RISK_ON
    if mkt < off.mkt and breadth_value < off.breadth:
        return RISK_OFF

    if prev_regime == RISK_ON and (mkt < on.mkt - hyst or breadth_value < on.breadth - hyst):
        return NEUTRAL
    if prev_regime == RISK_OFF and (mkt > off.mkt + hyst or breadth_value > off.breadth + hyst):
        return NEUTRAL

    return prev_regime


def gate_crypto(signal: float, regime: str, regime_cfg: RegimeConfig, shorts_enabled: bool) -> float:
    """Per-asset gate for crypto (and crypto-beta stocks COIN/MSTR/CRCL -- caller decides which gate
    to use per asset class using symbols.yaml's crypto_beta_stocks/bstocks lists; crypto-beta stocks
    use this gate, not gate_diversifier). BUILD_SPEC Section 6.3.

    Returns s_eff = signal * budget, or 0 if the signal doesn't clear the regime's entry threshold.
    NaN signal (insufficient history) propagates to NaN here; the caller treats NaN as weight 0.
    """
    if signal != signal:  # NaN check without importing math; NaN != NaN
        return float("nan")
    budget = regime_cfg.budgets[regime]
    if signal > budget.long_entry_min:
        return signal * budget.long
    if signal < budget.short_entry_max and shorts_enabled:
        return signal * budget.short
    return 0.0


def gate_diversifier(signal: float, diversifiers_cfg: DiversifiersConfig) -> float:
    """Per-asset gate for diversifiers (PAXG, bStocks): ignores the crypto regime entirely.
    Long-only unless diversifiers_cfg.allow_short. BUILD_SPEC Section 6.3."""
    if signal != signal:  # NaN check
        return float("nan")
    if signal > diversifiers_cfg.entry_min:
        return signal
    if diversifiers_cfg.allow_short and signal < -diversifiers_cfg.entry_min:
        return signal
    return 0.0
