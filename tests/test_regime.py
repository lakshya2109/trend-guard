"""Tests for trend_guard.strategy.regime: market index/signal, breadth, hysteresis, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import (
    DiversifiersConfig,
    RegimeBudget,
    RegimeConfig,
    RegimeThreshold,
    SignalConfig,
    VolConfig,
)
from trend_guard.strategy.regime import (
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    breadth,
    build_market_index,
    gate_crypto,
    gate_diversifier,
    market_signal,
    next_regime,
)

REGIME_CFG = RegimeConfig(
    index_size=3,
    on=RegimeThreshold(mkt=0.20, breadth=0.55),
    off=RegimeThreshold(mkt=-0.20, breadth=0.40),
    hysteresis=0.10,
    budgets={
        "RISK_ON": RegimeBudget(long=1.0, short=0.25, long_entry_min=0.10, short_entry_max=-0.50),
        "NEUTRAL": RegimeBudget(long=0.5, short=0.50, long_entry_min=0.20, short_entry_max=-0.30),
        "RISK_OFF": RegimeBudget(long=0.0, short=1.00, long_entry_min=1.01, short_entry_max=-0.30),
    },
)


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _trend_close(n: int, drift: float) -> pd.Series:
    log_price = np.cumsum(np.full(n, drift))
    return pd.Series(100.0 * np.exp(log_price), index=_hourly_index(n))


# --- breadth ---------------------------------------------------------------


def test_breadth_all_positive():
    assert breadth({"BTC": 0.5, "ETH": 0.3, "SOL": 0.1}) == 1.0


def test_breadth_none_positive():
    assert breadth({"BTC": -0.5, "ETH": -0.3}) == 0.0


def test_breadth_mixed():
    assert breadth({"BTC": 0.5, "ETH": -0.3, "SOL": 0.1, "XRP": -0.1}) == 0.5


def test_breadth_empty():
    assert breadth({}) == 0.0


def test_breadth_nan_signal_excluded():
    assert breadth({"BTC": 0.5, "ETH": float("nan")}) == 0.5


# --- build_market_index / market_signal ------------------------------------


def test_build_market_index_uptrend_rises():
    closes = {"BTC": _trend_close(100, 0.001), "ETH": _trend_close(100, 0.0015)}
    idx = build_market_index(closes)
    assert idx.iloc[-1] > idx.iloc[0]


def test_market_signal_uptrend_positive():
    n = max([24, 48, 96, 168, 336, 504]) + 300
    closes = {"BTC": _trend_close(n, 0.002), "ETH": _trend_close(n, 0.0025), "SOL": _trend_close(n, 0.0018)}
    signal_cfg = SignalConfig(
        lookbacks_h=[24, 48, 96, 168, 336, 504], w_donchian=0.5, w_tsmom=0.5, smooth_bars=3
    )
    vol_cfg = VolConfig(halflife_h=48, vol_floor_ann=0.15)
    mkt = market_signal(closes, signal_cfg, vol_cfg)
    assert mkt.iloc[-1] > 0.5


# --- next_regime hysteresis --------------------------------------------------


def test_next_regime_enters_risk_on():
    assert next_regime(NEUTRAL, mkt=0.25, breadth_value=0.60, regime_cfg=REGIME_CFG) == RISK_ON


def test_next_regime_enters_risk_off():
    assert next_regime(NEUTRAL, mkt=-0.25, breadth_value=0.30, regime_cfg=REGIME_CFG) == RISK_OFF


def test_next_regime_stays_risk_on_in_hysteresis_band():
    # mkt drops below on.mkt (0.20) but not below on.mkt - hyst (0.10) -> stays RISK_ON, no flip-flop
    assert next_regime(RISK_ON, mkt=0.15, breadth_value=0.60, regime_cfg=REGIME_CFG) == RISK_ON


def test_next_regime_exits_risk_on_to_neutral_below_hysteresis_band():
    assert next_regime(RISK_ON, mkt=0.05, breadth_value=0.60, regime_cfg=REGIME_CFG) == NEUTRAL


def test_next_regime_exits_risk_on_to_neutral_via_breadth():
    assert next_regime(RISK_ON, mkt=0.25, breadth_value=0.40, regime_cfg=REGIME_CFG) == NEUTRAL


def test_next_regime_stays_risk_off_in_hysteresis_band():
    # mkt rises above off.mkt (-0.20) but not above off.mkt + hyst (-0.10) -> stays RISK_OFF
    assert next_regime(RISK_OFF, mkt=-0.15, breadth_value=0.30, regime_cfg=REGIME_CFG) == RISK_OFF


def test_next_regime_exits_risk_off_to_neutral_above_hysteresis_band():
    assert next_regime(RISK_OFF, mkt=-0.05, breadth_value=0.30, regime_cfg=REGIME_CFG) == NEUTRAL


def test_next_regime_no_flip_flop_across_repeated_boundary_crossings():
    # Simulate mkt oscillating right at the on.mkt threshold: without hysteresis this would flip
    # every bar; with hysteresis it should stay in RISK_ON throughout since it never drops below
    # on.mkt - hyst.
    regime = RISK_ON
    mkt_path = [0.21, 0.19, 0.205, 0.195, 0.20, 0.199, 0.21]
    for mkt in mkt_path:
        regime = next_regime(regime, mkt=mkt, breadth_value=0.60, regime_cfg=REGIME_CFG)
        assert regime == RISK_ON


def test_next_regime_stays_neutral_when_no_condition_met():
    assert next_regime(NEUTRAL, mkt=0.0, breadth_value=0.45, regime_cfg=REGIME_CFG) == NEUTRAL


# --- gate_crypto -------------------------------------------------------------


def test_gate_crypto_long_entry_risk_on():
    s_eff = gate_crypto(0.5, RISK_ON, REGIME_CFG, shorts_enabled=True)
    assert s_eff == 0.5 * 1.0


def test_gate_crypto_below_long_entry_min_and_above_short_entry_max_is_zero():
    s_eff = gate_crypto(0.05, RISK_ON, REGIME_CFG, shorts_enabled=True)  # 0.05 < 0.10 entry_min
    assert s_eff == 0.0


def test_gate_crypto_short_entry_risk_on():
    s_eff = gate_crypto(-0.6, RISK_ON, REGIME_CFG, shorts_enabled=True)
    assert s_eff == -0.6 * 0.25


def test_gate_crypto_short_blocked_when_shorts_disabled():
    s_eff = gate_crypto(-0.6, RISK_ON, REGIME_CFG, shorts_enabled=False)
    assert s_eff == 0.0


def test_gate_crypto_risk_off_long_impossible():
    # long_entry_min is 1.01 in RISK_OFF budget -> no signal can ever enter long
    s_eff = gate_crypto(0.99, RISK_OFF, REGIME_CFG, shorts_enabled=True)
    assert s_eff == 0.0


def test_gate_crypto_risk_off_short_entry():
    s_eff = gate_crypto(-0.5, RISK_OFF, REGIME_CFG, shorts_enabled=True)
    assert s_eff == -0.5 * 1.00


def test_gate_crypto_nan_propagates():
    s_eff = gate_crypto(float("nan"), RISK_ON, REGIME_CFG, shorts_enabled=True)
    assert s_eff != s_eff  # NaN


# --- gate_diversifier ---------------------------------------------------------


def test_gate_diversifier_long_entry():
    cfg = DiversifiersConfig(entry_min=0.10, allow_short=False)
    assert gate_diversifier(0.3, cfg) == 0.3


def test_gate_diversifier_below_entry_min_is_zero():
    cfg = DiversifiersConfig(entry_min=0.10, allow_short=False)
    assert gate_diversifier(0.05, cfg) == 0.0


def test_gate_diversifier_negative_blocked_when_long_only():
    cfg = DiversifiersConfig(entry_min=0.10, allow_short=False)
    assert gate_diversifier(-0.5, cfg) == 0.0


def test_gate_diversifier_allows_short_when_configured():
    cfg = DiversifiersConfig(entry_min=0.10, allow_short=True)
    assert gate_diversifier(-0.5, cfg) == -0.5
