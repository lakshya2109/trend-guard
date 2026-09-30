"""strategy/book.py: causality, no leverage, eligibility filters, and the short-leg switch."""

import numpy as np
import pandas as pd
import pytest

from trend_guard.config import load_strategy_config
from trend_guard.strategy.book import book_weights, features, momentum_sleeve, trend_sleeve, tstat_score

CFG = load_strategy_config("config/strategy.yaml")


def synthetic_panel(n_days: int = 90, n_assets: int = 16, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assets with persistent but different drifts so both sleeves have something to do."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=24 * n_days, freq="1h", tz="UTC")
    drifts = np.linspace(-0.0006, 0.0008, n_assets)
    lr = drifts + rng.normal(0, 0.006, (len(idx), n_assets))
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(lr, axis=0)), index=idx, columns=[f"C{i}/USD" for i in range(n_assets)]
    )
    close["PAXG/USD"] = 2000 * np.exp(np.cumsum(0.0001 + rng.normal(0, 0.002, len(idx))))
    qv = pd.DataFrame(1e7, index=idx, columns=close.columns)  # $240M/day: every name is liquid
    return close, qv


def test_book_is_causal():
    close, qv = synthetic_panel()
    full = book_weights(close, qv, CFG.book)
    cut = book_weights(close.iloc[:-200], qv.iloc[:-200], CFG.book)
    # a row may only depend on bars up to that row: truncating the future cannot change the past
    np.testing.assert_allclose(full.iloc[:-200].to_numpy(), cut.to_numpy(), atol=1e-12)


def test_book_never_levers_and_trades_both_sides():
    close, qv = synthetic_panel()
    W = book_weights(close, qv, CFG.book).to_numpy()
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-9
    assert (W > 0).any() and (W < 0).any()


def test_trend_sleeve_is_long_only_and_momentum_excludes_paxg():
    close, qv = synthetic_panel()
    tr = trend_sleeve(close, qv, CFG.book.trend)
    mo = momentum_sleeve(close, qv, CFG.book.momentum)
    assert (tr >= 0).all()
    assert (mo[:, close.columns.get_loc("PAXG/USD")] == 0).all()
    assert np.abs(tr).sum(axis=1).max() <= CFG.book.trend.gross_cap + 1e-9
    assert np.abs(mo).sum(axis=1).max() <= CFG.book.momentum.gross_cap + 1e-9


def test_shorts_disabled_means_no_negative_weights():
    close, qv = synthetic_panel()
    assert (book_weights(close, qv, CFG.book, allow_short=False).to_numpy() >= 0).all()


def test_illiquid_and_young_coins_are_never_held():
    close, qv = synthetic_panel()
    qv["C15/USD"] = 1.0  # the strongest uptrend, but untradeable volume
    close.loc[close.index[: 24 * 80], "C14/USD"] = np.nan  # listed 10 days before the end
    W = book_weights(close, qv, CFG.book)
    assert (W["C15/USD"] == 0).all()
    assert (W["C14/USD"] == 0).all()  # younger than both sleeves' minimum age


def test_tstat_score_scales_with_trend_over_noise():
    close, qv = synthetic_panel()
    f = features(close, qv, "rolling", 168)
    sc = tstat_score(f, [168, 336])
    last = pd.Series(sc[-1], index=close.columns).drop("PAXG/USD")
    assert last.iloc[-1] > last.iloc[0]  # strongest drift scores above the weakest


def test_shares_above_one_are_rejected():
    close, qv = synthetic_panel(30, 4)
    bad = CFG.book.model_copy(update={"trend_share": 0.8, "momentum_share": 0.4})
    with pytest.raises(AssertionError):
        book_weights(close, qv, bad)


def test_team_book_is_long_short_capped_and_causal():
    from trend_guard.strategy.team_book import team_weights

    close, qv = synthetic_panel()
    W = team_weights(close, qv, CFG.book.team)
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-9
    assert (W[:, close.columns.get_loc("PAXG/USD")] >= 0).all()  # the team trend book is long-only
    cut = team_weights(close.iloc[:-200], qv.iloc[:-200], CFG.book.team)
    np.testing.assert_allclose(W[:-200], cut, atol=1e-12)


def test_blend_is_the_share_weighted_sum_of_its_parts():
    close, qv = synthetic_panel()
    b = CFG.book.model_copy(update={"risk_multiplier": 1.0})
    full = book_weights(close, qv, b).to_numpy()
    v2 = book_weights(close, qv, b.model_copy(update={"v2_share": 1.0, "team_share": 0.0})).to_numpy()
    team = book_weights(close, qv, b.model_copy(update={"v2_share": 0.0, "team_share": 1.0})).to_numpy()
    np.testing.assert_allclose(full, b.v2_share * v2 + b.team_share * team, atol=1e-12)


def test_risk_multiplier_scales_the_book_but_never_levers():
    close, qv = synthetic_panel()
    one = book_weights(close, qv, CFG.book.model_copy(update={"risk_multiplier": 1.0})).to_numpy()
    two = book_weights(close, qv, CFG.book.model_copy(update={"risk_multiplier": 2.0})).to_numpy()
    small = np.abs(one).sum(axis=1) <= 0.5  # rows the cap does not touch at x2
    np.testing.assert_allclose(two[small], 2 * one[small], atol=1e-12)
    huge = book_weights(close, qv, CFG.book.model_copy(update={"risk_multiplier": 50.0})).to_numpy()
    assert np.abs(huge).sum(axis=1).max() <= CFG.book.max_gross + 1e-9
