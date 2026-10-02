"""strategy/book.py's _cov_scale shrinkage parameter (TrendConfig.cov_shrink -- local research
experiment). Not part of the committed repo."""

from __future__ import annotations

import numpy as np

from trend_guard.strategy.book import _cov_scale, _weighted_cov


def test_shrink_zero_matches_original_unshrunk_behaviour():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (168, 5))
    w = np.array([0.2, -0.1, 0.15, 0.0, 0.05])
    base = _cov_scale(w, rets, 0.01, 1.0)
    explicit_zero = _cov_scale(w, rets, 0.01, 1.0, shrink=0.0)
    assert base == explicit_zero


def test_shrink_one_matches_a_hand_built_constant_correlation_covariance():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, (168, 3))
    w = np.array([0.3, 0.2, -0.1])
    scale = _cov_scale(w, rets, 0.01, 1.0, shrink=1.0, rho=0.7)

    C_sample = np.cov(rets, rowvar=False) * 24
    std = np.sqrt(np.diag(C_sample))
    target = 0.7 * np.outer(std, std)
    np.fill_diagonal(target, np.diag(C_sample))
    vol = np.sqrt(w @ target @ w)
    expected = min(0.01 / vol, 1.0 / np.abs(w).sum())
    assert abs(scale - expected) < 1e-12


def test_shrinkage_changes_sizing_when_sample_covariance_is_noisy():
    """With few observations relative to the number of held names, the sample covariance is
    unstable -- shrinking toward the constant-correlation target should give a materially
    different (and, for a near-singular sample matrix, a better-conditioned) scale factor."""
    rng = np.random.default_rng(2)
    n_obs, n_assets = 20, 15  # deliberately few observations relative to names: a noisy regime
    rets = rng.normal(0, 0.01, (n_obs, n_assets))
    w = rng.uniform(0.01, 0.05, n_assets)
    unshrunk = _cov_scale(w, rets, 0.01, 1.0, shrink=0.0)
    shrunk = _cov_scale(w, rets, 0.01, 1.0, shrink=0.5, rho=0.7)
    assert unshrunk != shrunk
    assert np.isfinite(unshrunk) and np.isfinite(shrunk)


def test_weighted_cov_with_uniform_weights_matches_numpy_cov():
    rng = np.random.default_rng(3)
    R = rng.normal(0, 0.01, (50, 4))
    uniform = np.full(50, 1.0 / 50)
    got = _weighted_cov(R, uniform)
    expected = np.cov(R, rowvar=False)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_ewma_halflife_zero_matches_equal_weighted_behaviour():
    rng = np.random.default_rng(4)
    rets = rng.normal(0, 0.01, (336, 6))
    w = rng.uniform(0.01, 0.05, 6)
    base = _cov_scale(w, rets, 0.004, 1.0, shrink=0.1, rho=0.7)
    explicit = _cov_scale(w, rets, 0.004, 1.0, shrink=0.1, rho=0.7, ewma_halflife=0.0)
    assert base == explicit


def test_ewma_halflife_reacts_faster_to_a_recent_vol_pickup():
    """A window that's calm for its first half and much more volatile in its second half: the
    EWMA-weighted covariance (recent data counts more) should estimate a higher variance than the
    equal-weighted one, which still averages in the calm first half -- exactly the mechanism meant
    to make a long cov_hours window track a fresh vol pickup instead of smoothing it away."""
    rng = np.random.default_rng(5)
    n = 336
    calm = rng.normal(0, 0.002, (n // 2, 3))
    volatile = rng.normal(0, 0.02, (n - n // 2, 3))
    rets = np.vstack([calm, volatile])  # calm first, volatile most recent (last rows)
    w = np.array([0.1, 0.1, 0.1])
    equal_weighted = _cov_scale(w, rets, 0.004, 1.0)
    ewma = _cov_scale(w, rets, 0.004, 1.0, ewma_halflife=48.0)
    # a higher estimated vol -> a *smaller* scale factor (sizes down more)
    assert ewma < equal_weighted
