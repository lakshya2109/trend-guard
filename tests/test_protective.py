"""Tests for trend_guard.risk.protective: ATR, trailing stops, shock filter, cooldown."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from trend_guard.config import ProtectiveConfig
from trend_guard.risk.protective import (
    atr,
    cooldown_active,
    shock_filter,
    trailing_stop_long,
    trailing_stop_short,
)

PROTECTIVE_CFG = ProtectiveConfig(atr_mult=2.5, shock_z=2.5, squeeze_prior_drop=-0.08, cooldown_h=6)


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


# --- atr ---------------------------------------------------------------


def test_atr_constant_price_is_zero():
    n = 30
    close = pd.Series([100.0] * n, index=_hourly_index(n))
    high = close.copy()
    low = close.copy()
    result = atr(high, low, close, window=24)
    assert result.iloc[-1] == 0.0


def test_atr_positive_for_ranging_bars():
    n = 30
    idx = _hourly_index(n)
    close = pd.Series([100.0] * n, index=idx)
    high = pd.Series([101.0] * n, index=idx)
    low = pd.Series([99.0] * n, index=idx)
    result = atr(high, low, close, window=24)
    assert result.iloc[-1] == 2.0  # true range is constant 2.0 (high-low), no gap vs prev close


def test_atr_nan_before_window_full():
    n = 10
    close = pd.Series(range(100, 110), index=_hourly_index(n)).astype(float)
    high = close + 1
    low = close - 1
    result = atr(high, low, close, window=24)
    assert result.isna().all()


# --- trailing_stop_long / short -----------------------------------------


def test_trailing_stop_long_exits_below_atr_stop():
    # best=110, atr=2, atr_mult=2.5 -> stop level = max(110 - 5, mid_ref) = max(105, 100) = 105
    exit_ = trailing_stop_long(
        price=104.0,
        best_price_since_entry=110.0,
        atr_value=2.0,
        mid_channel_ref=100.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is True


def test_trailing_stop_long_holds_above_atr_stop():
    exit_ = trailing_stop_long(
        price=106.0,
        best_price_since_entry=110.0,
        atr_value=2.0,
        mid_channel_ref=100.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is False


def test_trailing_stop_long_uses_mid_channel_ref_when_higher():
    # atr-based stop = 110 - 5 = 105, but mid_channel_ref=108 is higher -> effective stop is 108
    exit_ = trailing_stop_long(
        price=107.0,
        best_price_since_entry=110.0,
        atr_value=2.0,
        mid_channel_ref=108.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is True


def test_trailing_stop_short_exits_above_atr_stop():
    # best (lowest since entry)=90, atr=2 -> stop level = min(90+5, mid_ref) = min(95, 100) = 95
    exit_ = trailing_stop_short(
        price=96.0,
        best_price_since_entry=90.0,
        atr_value=2.0,
        mid_channel_ref=100.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is True


def test_trailing_stop_short_holds_below_atr_stop():
    exit_ = trailing_stop_short(
        price=94.0,
        best_price_since_entry=90.0,
        atr_value=2.0,
        mid_channel_ref=100.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is False


def test_trailing_stop_short_uses_mid_channel_ref_when_lower():
    exit_ = trailing_stop_short(
        price=93.0,
        best_price_since_entry=90.0,
        atr_value=2.0,
        mid_channel_ref=92.0,
        protective_cfg=PROTECTIVE_CFG,
    )
    assert exit_ is True


# --- shock_filter -----------------------------------------------------------


def test_shock_filter_no_action_when_calm():
    result = shock_filter(r15=0.0, sigma_h=0.01, protective_cfg=PROTECTIVE_CFG, btc_72h_return=0.0)
    assert result["cut_longs_pct"] == 0.0
    assert result["cover_shorts_pct"] == 0.0


def test_shock_filter_cuts_longs_on_crash():
    # z = r15 / (sigma_h * sqrt(0.25)); pick r15 deeply negative to exceed -shock_z
    sigma_h = 0.01
    r15 = -0.5 * sigma_h  # z = -0.5/0.5 = -1... need bigger; recompute below
    # sqrt(0.25) = 0.5, so z = r15 / (sigma_h*0.5). Need z < -2.5 -> r15 < -2.5*0.5*sigma_h = -1.25*sigma_h
    r15 = -1.3 * sigma_h
    result = shock_filter(r15=r15, sigma_h=sigma_h, protective_cfg=PROTECTIVE_CFG, btc_72h_return=0.0)
    assert result["cut_longs_pct"] == 0.5
    assert result["freeze_longs_until_min"] == 60
    assert result["cover_shorts_pct"] == 0.0


def test_shock_filter_covers_shorts_on_squeeze():
    sigma_h = 0.01
    r15 = 1.3 * sigma_h  # z > +2.5
    result = shock_filter(r15=r15, sigma_h=sigma_h, protective_cfg=PROTECTIVE_CFG, btc_72h_return=-0.10)
    assert result["cover_shorts_pct"] == 0.5
    assert result["freeze_shorts_until_min"] == 60
    assert result["cut_longs_pct"] == 0.0


def test_shock_filter_no_squeeze_cover_without_prior_drop():
    sigma_h = 0.01
    r15 = 1.3 * sigma_h  # z > +2.5, but btc_72h_return not deep enough
    result = shock_filter(r15=r15, sigma_h=sigma_h, protective_cfg=PROTECTIVE_CFG, btc_72h_return=-0.01)
    assert result["cover_shorts_pct"] == 0.0


def test_shock_filter_zero_sigma_is_no_op():
    result = shock_filter(r15=-0.5, sigma_h=0.0, protective_cfg=PROTECTIVE_CFG, btc_72h_return=-0.10)
    assert result["cut_longs_pct"] == 0.0
    assert result["cover_shorts_pct"] == 0.0


# --- cooldown_active ----------------------------------------------------------


def test_cooldown_active_true_within_window():
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    cooldown_until = {"BTC/USD": now + timedelta(hours=2)}
    assert cooldown_active("BTC/USD", cooldown_until, now) is True


def test_cooldown_active_false_after_window():
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    cooldown_until = {"BTC/USD": now - timedelta(hours=1)}
    assert cooldown_active("BTC/USD", cooldown_until, now) is False


def test_cooldown_active_false_when_absent():
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert cooldown_active("ETH/USD", {}, now) is False
