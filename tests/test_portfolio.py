"""Tests for trend_guard.strategy.portfolio: sizing, caps, covariance, vol targeting, gross."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import PortfolioCaps, PortfolioConfig, VolConfig
from trend_guard.strategy.portfolio import (
    apply_aggregate_caps,
    apply_caps,
    build_portfolio,
    enforce_gross,
    portfolio_vol,
    raw_weight,
    scale_to_target,
    shrink_covariance,
    sigma_ann,
)
from trend_guard.strategy.signals import ANN_H

VOL_CFG = VolConfig(halflife_h=48, vol_floor_ann=0.15)
PORTFOLIO_CFG = PortfolioConfig(
    sigma_asset_target=0.15,
    sigma_portfolio_target=0.30,
    cov_halflife_h=72,
    shrink=0.30,
    k_max=3.0,
    gross_max=0.98,
    caps=PortfolioCaps(
        btc_eth=0.25, major=0.12, other=0.06, bstock=0.08, bstock_sleeve=0.25, paxg=0.20, short_total=0.50
    ),
)


def test_sigma_ann():
    hourly = pd.Series([0.01, 0.02, 0.03])
    result = sigma_ann(hourly)
    expected = hourly * np.sqrt(ANN_H)
    pd.testing.assert_series_equal(result, expected)


def test_raw_weight_basic():
    # s_eff=0.5, sigma_ann=0.30 (above floor) -> 0.5 * 0.15 / 0.30 = 0.25
    w = raw_weight(0.5, 0.30, VOL_CFG, PORTFOLIO_CFG)
    assert abs(w - 0.25) < 1e-9


def test_raw_weight_uses_vol_floor():
    # sigma_ann below floor (0.15) -> denominator clamped to floor
    w = raw_weight(0.5, 0.05, VOL_CFG, PORTFOLIO_CFG)
    expected = 0.5 * 0.15 / 0.15
    assert abs(w - expected) < 1e-9


def test_raw_weight_nan_propagates():
    w = raw_weight(float("nan"), 0.30, VOL_CFG, PORTFOLIO_CFG)
    assert w != w


# --- apply_caps ---------------------------------------------------------------


def test_apply_caps_clips_overweight_position():
    weights = {"BTC/USD": 0.9, "SOME_OTHER/USD": 0.5}
    bucket_of = {"BTC/USD": "btc_eth", "SOME_OTHER/USD": "other"}
    capped = apply_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert capped["BTC/USD"] == PORTFOLIO_CFG.caps.btc_eth
    assert capped["SOME_OTHER/USD"] == PORTFOLIO_CFG.caps.other


def test_apply_caps_clips_negative_overweight():
    weights = {"BTC/USD": -0.9}
    bucket_of = {"BTC/USD": "btc_eth"}
    capped = apply_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert capped["BTC/USD"] == -PORTFOLIO_CFG.caps.btc_eth


def test_apply_caps_leaves_within_cap_unchanged():
    weights = {"BTC/USD": 0.1}
    bucket_of = {"BTC/USD": "btc_eth"}
    capped = apply_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert capped["BTC/USD"] == 0.1


def test_apply_caps_unknown_bucket_passes_through():
    weights = {"WEIRD/USD": 0.9}
    capped = apply_caps(weights, {}, PORTFOLIO_CFG)
    assert capped["WEIRD/USD"] == 0.9


def test_apply_caps_nan_passthrough():
    weights = {"BTC/USD": float("nan")}
    bucket_of = {"BTC/USD": "btc_eth"}
    capped = apply_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert capped["BTC/USD"] != capped["BTC/USD"]


# --- apply_aggregate_caps ------------------------------------------------------


def test_apply_aggregate_caps_scales_shorts_pro_rata_preserving_proportions():
    # total short collateral 0.30 + 0.30 = 0.60 > cap 0.50 -> scale factor 0.50/0.60
    weights = {"A/USD": -0.30, "B/USD": -0.30, "C/USD": 0.10}
    bucket_of = {"A/USD": "other", "B/USD": "other", "C/USD": "other"}
    out = apply_aggregate_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert abs(out["A/USD"] - out["B/USD"]) < 1e-12  # proportions preserved (equal inputs -> equal outputs)
    total_short = -out["A/USD"] - out["B/USD"]
    assert abs(total_short - PORTFOLIO_CFG.caps.short_total) < 1e-9
    assert out["C/USD"] == 0.10  # long side untouched


def test_apply_aggregate_caps_preserves_relative_proportions_when_unequal():
    weights = {"A/USD": -0.40, "B/USD": -0.20}  # total 0.60, ratio 2:1
    bucket_of = {"A/USD": "other", "B/USD": "other"}
    out = apply_aggregate_caps(weights, bucket_of, PORTFOLIO_CFG)
    ratio = (-out["A/USD"]) / (-out["B/USD"])
    assert abs(ratio - 2.0) < 1e-9


def test_apply_aggregate_caps_no_scaling_when_under_cap():
    weights = {"A/USD": -0.10, "B/USD": -0.10}
    bucket_of = {"A/USD": "other", "B/USD": "other"}
    out = apply_aggregate_caps(weights, bucket_of, PORTFOLIO_CFG)
    assert out["A/USD"] == -0.10
    assert out["B/USD"] == -0.10


def test_apply_aggregate_caps_bstock_sleeve_scaled():
    weights = {"COIN/USD": 0.20, "MSTR/USD": 0.20}  # total 0.40 > cap 0.25
    bucket_of = {"COIN/USD": "bstock", "MSTR/USD": "bstock"}
    out = apply_aggregate_caps(weights, bucket_of, PORTFOLIO_CFG)
    total = out["COIN/USD"] + out["MSTR/USD"]
    assert abs(total - PORTFOLIO_CFG.caps.bstock_sleeve) < 1e-9
    assert abs(out["COIN/USD"] - out["MSTR/USD"]) < 1e-12


# --- portfolio_vol / shrink_covariance / scale_to_target -----------------------


def test_portfolio_vol_single_asset():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])  # variance 0.04 -> vol 0.2
    weights = {"A": 1.0}
    assert abs(portfolio_vol(weights, cov) - 0.2) < 1e-9


def test_shrink_covariance_full_shrink_is_diagonal():
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"])
    shrunk = shrink_covariance(cov, shrink=1.0)
    assert abs(shrunk.loc["A", "B"]) < 1e-12
    assert abs(shrunk.loc["A", "A"] - 0.04) < 1e-12


def test_shrink_covariance_zero_shrink_is_identity():
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"])
    shrunk = shrink_covariance(cov, shrink=0.0)
    pd.testing.assert_frame_equal(shrunk, cov)


def test_scale_to_target_hits_target_vol_when_not_capped_by_k_max():
    cov = pd.DataFrame([[0.01]], index=["A"], columns=["A"])  # vol 0.1
    weights = {"A": 1.0}  # port_vol = 0.1
    scaled = scale_to_target(weights, cov, PORTFOLIO_CFG)
    # k = min(0.30/0.1, 3.0) = min(3.0, 3.0) = 3.0 -> weight 3.0, port_vol becomes 0.30
    assert abs(scaled["A"] - 3.0) < 1e-9


def test_scale_to_target_clamped_by_k_max():
    cov = pd.DataFrame([[0.0001]], index=["A"], columns=["A"])  # vol 0.01, k would be 30 without cap
    weights = {"A": 1.0}
    scaled = scale_to_target(weights, cov, PORTFOLIO_CFG)
    assert abs(scaled["A"] - PORTFOLIO_CFG.k_max) < 1e-9


# --- enforce_gross --------------------------------------------------------------


def test_enforce_gross_scales_down_pro_rata_when_over_budget():
    weights = {"A": 0.6, "B": -0.6}  # gross 1.2 > 0.98
    out = enforce_gross(weights, PORTFOLIO_CFG.gross_max)
    gross = abs(out["A"]) + abs(out["B"])
    assert abs(gross - PORTFOLIO_CFG.gross_max) < 1e-9
    assert abs(out["A"] / out["B"] - (0.6 / -0.6)) < 1e-9  # proportions preserved


def test_enforce_gross_no_scaling_when_under_budget():
    weights = {"A": 0.3, "B": -0.2}
    out = enforce_gross(weights, PORTFOLIO_CFG.gross_max)
    assert out == weights


# --- build_portfolio orchestration ----------------------------------------------


def test_build_portfolio_end_to_end_respects_gross_and_caps():
    s_eff = {"BTC/USD": 0.9, "ETH/USD": 0.8, "SOL/USD": 0.9, "OTHER/USD": 0.9}
    sigma_ann_by_asset = {"BTC/USD": 0.4, "ETH/USD": 0.4, "SOL/USD": 0.5, "OTHER/USD": 0.6}
    assets = list(s_eff.keys())
    cov = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)
    bucket_of = {"BTC/USD": "btc_eth", "ETH/USD": "btc_eth", "SOL/USD": "major", "OTHER/USD": "other"}

    w = build_portfolio(s_eff, sigma_ann_by_asset, cov, bucket_of, VOL_CFG, PORTFOLIO_CFG)

    assert abs(w["BTC/USD"]) <= PORTFOLIO_CFG.caps.btc_eth + 1e-9
    assert abs(w["ETH/USD"]) <= PORTFOLIO_CFG.caps.btc_eth + 1e-9
    assert abs(w["SOL/USD"]) <= PORTFOLIO_CFG.caps.major + 1e-9
    assert abs(w["OTHER/USD"]) <= PORTFOLIO_CFG.caps.other + 1e-9
    gross = sum(abs(v) for v in w.values())
    assert gross <= PORTFOLIO_CFG.gross_max + 1e-9


def test_build_portfolio_nan_signal_yields_nan_weight():
    s_eff = {"BTC/USD": float("nan"), "ETH/USD": 0.5}
    sigma_ann_by_asset = {"BTC/USD": 0.4, "ETH/USD": 0.4}
    assets = ["BTC/USD", "ETH/USD"]
    cov = pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets)
    bucket_of = {"BTC/USD": "btc_eth", "ETH/USD": "btc_eth"}
    w = build_portfolio(s_eff, sigma_ann_by_asset, cov, bucket_of, VOL_CFG, PORTFOLIO_CFG)
    assert w["BTC/USD"] != w["BTC/USD"]
