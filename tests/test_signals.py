"""Tests for trend_guard.strategy.signals: ewma_vol and compute_signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import SignalConfig, VolConfig
from trend_guard.strategy.signals import ANN_H, compute_signal, ewma_vol

SIGNAL_CFG = SignalConfig(lookbacks_h=[24, 48, 96, 168, 336, 504], w_donchian=0.5, w_tsmom=0.5, smooth_bars=3)
VOL_CFG = VolConfig(halflife_h=48, vol_floor_ann=0.15)


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _uptrend_close(n: int, drift: float = 0.002, noise: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = np.full(n, drift) + (rng.normal(0, noise, n) if noise else 0.0)
    log_price = np.cumsum(steps)
    return pd.Series(100.0 * np.exp(log_price), index=_hourly_index(n))


def test_ann_h_constant():
    assert ANN_H == 24 * 365


def test_ewma_vol_zero_for_constant_price():
    close = pd.Series([100.0] * 100, index=_hourly_index(100))
    log_ret = np.log(close / close.shift(1))
    vol = ewma_vol(log_ret, halflife_h=48)
    # first value is NaN (log_ret[0] is NaN), rest should be exactly 0 for a flat series
    assert vol.iloc[1:].abs().max() < 1e-12


def test_ewma_vol_positive_for_noisy_series():
    close = _uptrend_close(200, drift=0.0, noise=0.01, seed=1)
    log_ret = np.log(close / close.shift(1))
    vol = ewma_vol(log_ret, halflife_h=48)
    assert vol.iloc[-1] > 0


def test_signal_nan_before_min_history():
    n_bars = max(SIGNAL_CFG.lookbacks_h)  # exactly at the boundary, one short of required
    close = _uptrend_close(n_bars)
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    assert sig.isna().all()


def test_signal_non_nan_after_min_history():
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 200
    close = _uptrend_close(n_bars)
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    min_bars = max(SIGNAL_CFG.lookbacks_h) + 1
    assert sig.iloc[: min_bars - 1].isna().all()
    assert sig.iloc[min_bars - 1 :].notna().all()


def test_strong_sustained_uptrend_signal_near_one():
    # Long strong uptrend with negligible noise: signal should approach +1 within the smoothing lag.
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 600
    close = _uptrend_close(n_bars, drift=0.003, noise=0.0)
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    assert sig.iloc[-1] > 0.85


def test_strong_sustained_downtrend_signal_near_minus_one():
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 600
    close = _uptrend_close(n_bars, drift=-0.003, noise=0.0)
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    assert sig.iloc[-1] < -0.85


def test_flat_price_signal_near_zero():
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 200
    close = pd.Series([100.0] * n_bars, index=_hourly_index(n_bars))
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    # flat price: Donchian mid == price (0), momentum log-ret == 0 -> signal exactly 0 (no NaN issues)
    assert abs(sig.iloc[-1]) < 1e-9


def test_random_walk_signal_near_zero_on_average():
    # Any single random walk can trend, so check symmetry: averaged over many driftless paths the
    # final signal should be close to zero (no directional bias built into the signal).
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 100
    finals = [
        compute_signal(_uptrend_close(n_bars, drift=0.0, noise=0.01, seed=s), SIGNAL_CFG, VOL_CFG).iloc[-1]
        for s in range(40)
    ]
    assert abs(float(np.mean(finals))) < 0.2


def test_signal_is_causal_matches_prefix_computation():
    # Computing the full series and slicing at t must equal computing on close[:t] directly
    # (no look-ahead leakage) -- this is required for backtest/live consistency.
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 300
    close = _uptrend_close(n_bars, drift=0.001, noise=0.005, seed=7)
    full_sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)

    cut = n_bars - 50
    prefix_sig = compute_signal(close.iloc[:cut], SIGNAL_CFG, VOL_CFG)

    pd.testing.assert_series_equal(full_sig.iloc[:cut], prefix_sig, check_names=False)


def test_signal_bounded_in_range():
    n_bars = max(SIGNAL_CFG.lookbacks_h) + 300
    close = _uptrend_close(n_bars, drift=0.001, noise=0.02, seed=3)
    sig = compute_signal(close, SIGNAL_CFG, VOL_CFG)
    valid = sig.dropna()
    assert (valid >= -1.0001).all()
    assert (valid <= 1.0001).all()
