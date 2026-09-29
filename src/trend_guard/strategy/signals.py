"""Trend ensemble signal: Donchian position + vol-normalised momentum, per BUILD_SPEC.md Section 6.1-6.2.

Pure functions only: no I/O, no clock reads. Every function is causal (uses only close[:t] for each t)
so the backtest engine can walk bar-by-bar with no future leakage, and the live engine can read the
last value of the returned series with no drift between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import SignalConfig, VolConfig

ANN_H = 24 * 365  # hours per year, used to annualise hourly vol elsewhere (portfolio.py)
_EPS = 1e-8


def ewma_vol(log_ret: pd.Series, halflife_h: int) -> pd.Series:
    """EWMA standard deviation of hourly log returns (not annualised). BUILD_SPEC Section 6.1."""
    return log_ret.pow(2).ewm(halflife=halflife_h, adjust=False).mean().pow(0.5)


def _donchian_position(close: pd.Series, lookback_h: int) -> pd.Series:
    """d_L: clipped position of close within its L-bar Donchian channel (causal rolling window)."""
    upper = close.rolling(window=lookback_h, min_periods=lookback_h).max()
    lower = close.rolling(window=lookback_h, min_periods=lookback_h).min()
    mid = (upper + lower) / 2.0
    raw = (close - mid) / (0.5 * (upper - lower) + _EPS)
    return raw.clip(lower=-1.0, upper=1.0)


def _vol_norm_momentum(close: pd.Series, sigma_h: pd.Series, lookback_h: int) -> pd.Series:
    """m_L: tanh of vol-normalised log return over L bars, causal."""
    log_ret_l = np.log(close / close.shift(lookback_h))
    denom = sigma_h * np.sqrt(lookback_h) + _EPS
    return np.tanh(log_ret_l / denom)


def compute_signal(close: pd.Series, signal_cfg: SignalConfig, vol_cfg: VolConfig) -> pd.Series:
    """Full ensemble trend signal, per BUILD_SPEC Section 6.2.

    `close` is a single asset's hourly close-price series indexed by UTC hour, in chronological order.
    Returns a series aligned to `close.index`: NaN wherever the asset has fewer than
    `max(lookbacks_h) + 1` bars of history (per spec, NaN means "target weight 0" downstream).

    Causal by construction (rolling/shift/ewm all look backward only), so the same series computed
    once over full history and sliced at [:t] equals the series computed on close[:t] alone -- the
    backtest engine may call this once per asset and index into it bar-by-bar without recomputation
    drift, or the live engine may call it fresh each cycle and take .iloc[-1].
    """
    log_ret = np.log(close / close.shift(1))
    sigma_h = ewma_vol(log_ret, vol_cfg.halflife_h)

    components = []
    for lookback_h in signal_cfg.lookbacks_h:
        d_l = _donchian_position(close, lookback_h)
        m_l = _vol_norm_momentum(close, sigma_h, lookback_h)
        components.append(signal_cfg.w_donchian * d_l + signal_cfg.w_tsmom * m_l)

    signal_raw = pd.concat(components, axis=1).mean(axis=1)

    # Mask to NaN before smoothing (not after): a per-lookback row mean can be non-NaN even when
    # some longer lookbacks aren't ready yet (pandas skips NaN in the mean), so without this the
    # EMA state would get seeded by premature partial averages. Masking first makes ewm's leading-
    # NaN skip start the recursion cleanly at the first bar with full history.
    min_bars = max(signal_cfg.lookbacks_h) + 1
    bars_seen = np.arange(1, len(close) + 1)
    insufficient_history = pd.Series(bars_seen < min_bars, index=close.index)
    signal_raw = signal_raw.mask(insufficient_history)

    signal = signal_raw.ewm(span=signal_cfg.smooth_bars, adjust=False).mean()
    return signal
