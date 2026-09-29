"""Metrics tests. BUILD_SPEC.md Section 11.1."""

import numpy as np
import pandas as pd

from trend_guard.metrics import all_definitions, daily_ann, daily_raw, hourly_raw, turnover


def test_daily_raw_matches_worked_example_path_a():
    """Path A from roostoo_hackathon_research_report.md Section 2.1: steady +0.3%/day, two -0.2% days."""
    returns = pd.Series([0.003] * 12 + [-0.002] * 2)
    result = daily_raw(returns)
    assert result.total_return > 0.03
    assert result.composite > 1.0  # report's worked example gives ~1.77 under this definition


def test_daily_raw_single_bad_day_tanks_composite():
    """Path C: same as A but one day is -2% -- composite should collapse."""
    steady = [0.003] * 13 + [-0.002]
    bad = [0.003] * 12 + [-0.002, -0.02]
    steady_composite = daily_raw(pd.Series(steady)).composite
    bad_composite = daily_raw(pd.Series(bad)).composite
    assert bad_composite < steady_composite * 0.5


def test_max_drawdown_zero_for_never_red():
    returns = pd.Series([0.001] * 14)
    result = daily_raw(returns)
    assert result.max_drawdown == 0.0
    assert result.denom_hit_eps is True  # calmar denominator clamped at eps, degenerate


def test_daily_ann_scales_by_sqrt_365():
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 100))
    raw = daily_raw(returns)
    ann = daily_ann(returns)
    assert ann.sharpe == raw.sharpe * np.sqrt(365)


def test_red_share_counts_negative_days():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    result = daily_raw(returns)
    assert result.red_share == 0.4


def test_hourly_raw_runs_on_hourly_series():
    returns = pd.Series(np.random.default_rng(1).normal(0.0001, 0.002, 24 * 14))
    result = hourly_raw(returns)
    assert isinstance(result.composite, float)


def test_turnover_zero_when_no_trading():
    notional = pd.Series([0.0, 0.0, 0.0])
    equity = pd.Series([100_000.0, 100_000.0, 100_000.0])
    assert turnover(notional, equity) == 0.0


def test_turnover_positive_when_trading():
    notional = pd.Series([1000.0, 2000.0])
    equity = pd.Series([100_000.0, 100_000.0])
    assert turnover(notional, equity) > 0


def test_all_definitions_returns_three_keys():
    daily = pd.Series([0.001] * 14)
    hourly = pd.Series([0.00004] * (24 * 14))
    results = all_definitions(daily, hourly)
    assert set(results.keys()) == {"daily_raw", "daily_ann", "hourly_raw"}
