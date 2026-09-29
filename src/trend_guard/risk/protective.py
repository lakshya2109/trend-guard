"""Protective loop primitives: ATR, trailing stops, shock filter, cooldown check.

BUILD_SPEC.md Section 7.1. Pure functions only -- this module never touches the broker. It returns
structured decisions (bool exit flags, or a dict of actions for the shock filter); the caller (the
live engine's 60s protective loop, or the backtest engine's hourly high/low evaluation) executes them.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from trend_guard.config import ProtectiveConfig


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24) -> pd.Series:
    """Mean true range over `window` bars. BUILD_SPEC Section 7.1 (default window: last 24 hourly bars).

    True range = max(high-low, |high - prev_close|, |low - prev_close|). ATR = rolling mean of TR.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def trailing_stop_long(
    price: float,
    best_price_since_entry: float,
    atr_value: float,
    mid_channel_ref: float,
    protective_cfg: ProtectiveConfig,
) -> bool:
    """True = exit at MARKET. Long stop level = max(best_price_since_entry - atr_mult*ATR,
    mid_channel_ref), ratcheted in the trade's favour only (caller must pass a `best_price_since_entry`
    that only ever increases). BUILD_SPEC Section 7.1."""
    stop_level = max(best_price_since_entry - protective_cfg.atr_mult * atr_value, mid_channel_ref)
    return price < stop_level


def trailing_stop_short(
    price: float,
    best_price_since_entry: float,
    atr_value: float,
    mid_channel_ref: float,
    protective_cfg: ProtectiveConfig,
) -> bool:
    """True = exit at MARKET. Mirror of trailing_stop_long: short stop level = min(best_price_since_entry
    + atr_mult*ATR, mid_channel_ref), where `best_price_since_entry` is the lowest price seen since
    entry (caller must pass a value that only ever decreases). BUILD_SPEC Section 7.1."""
    stop_level = min(best_price_since_entry + protective_cfg.atr_mult * atr_value, mid_channel_ref)
    return price > stop_level


def shock_filter(r15: float, sigma_h: float, protective_cfg: ProtectiveConfig, btc_72h_return: float) -> dict:
    """Shock/squeeze filter. BUILD_SPEC Section 7.1: z = r15 / (sigma_h * sqrt(0.25)).

    z < -shock_z: cut all crypto longs by 50%, freeze new crypto longs for 60 min.
    z > +shock_z and btc_72h_return < squeeze_prior_drop: cover 50% of shorts, freeze new shorts 60 min.

    Returns a structured dict of the resulting action; this function never touches the broker.
    The `0.25` and the `50%` cuts and the `60` minute freeze are BUILD_SPEC formula literals (Section
    7.1 text), not tunable config thresholds, so they appear as literals here matching the spec text.
    """
    result = {
        "cut_longs_pct": 0.0,
        "freeze_longs_until_min": 0,
        "cover_shorts_pct": 0.0,
        "freeze_shorts_until_min": 0,
    }
    if sigma_h <= 0:
        return result  # no volatility estimate available, no-op

    z = r15 / (sigma_h * math.sqrt(0.25))

    if z < -protective_cfg.shock_z:
        result["cut_longs_pct"] = 0.5
        result["freeze_longs_until_min"] = 60
    if z > protective_cfg.shock_z and btc_72h_return < protective_cfg.squeeze_prior_drop:
        result["cover_shorts_pct"] = 0.5
        result["freeze_shorts_until_min"] = 60

    return result


def cooldown_active(asset: str, cooldown_until: dict[str, datetime], now: datetime) -> bool:
    """True if `asset` is still within its post-stop-out cooldown window. The cooldown *state* (when
    a stop-out happened, and thus when the cooldown ends) is tracked by the caller/state store; this
    function only compares `now` against the given `cooldown_until` map. BUILD_SPEC Section 7.1."""
    until = cooldown_until.get(asset)
    if until is None:
        return False
    return now < until
