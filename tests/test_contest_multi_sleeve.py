"""backtest/contest.py: simulate_multi_sleeve -- per-sleeve drawdown breaker (local research
experiment, see docs/STRATEGY.md research log). Not part of the committed repo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.contest import simulate, simulate_multi_sleeve, window_starts
from trend_guard.config import BacktestConfig, BreakerConfig, ExecutionConfig

BT = BacktestConfig(
    first_window="2025-01-01", last_window="2025-01-01", window_days=2, fee=0.001, slippage=0.0
)
EX = ExecutionConfig()
NO_BRAKE = BreakerConfig(dd_level=1.0)


def flat_close(hours: int = 24 * 5, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame({"A/USD": price, "B/USD": price}, index=idx)


def test_matches_simulate_when_the_breaker_never_trips():
    """With dd_level so high it can never fire, per-sleeve breaking is a no-op and
    simulate_multi_sleeve on the split sleeves must reproduce simulate on their sum exactly."""
    close = flat_close()
    close.iloc[20:] = 110.0
    sleeve_a = np.full((len(close), 2), 0.0)
    sleeve_a[:, 0] = 0.3
    sleeve_b = np.full((len(close), 2), 0.0)
    sleeve_b[:, 1] = -0.2
    combined = sleeve_a + sleeve_b
    starts = window_starts(close.index, BT)

    eq_single, turn_single = simulate(combined, close, starts, BT, EX, NO_BRAKE)
    eq_multi, turn_multi = simulate_multi_sleeve(
        {"a": sleeve_a, "b": sleeve_b}, close, starts, BT, EX, NO_BRAKE
    )
    np.testing.assert_allclose(eq_single, eq_multi, rtol=1e-9)
    np.testing.assert_allclose(turn_single, turn_multi, rtol=1e-9)


def test_only_the_crashing_sleeve_gets_braked():
    """Sleeve A's own book (long A/USD) crashes mid-window; sleeve B's own book (long a flat
    asset) does not. The per-sleeve breaker must cut sleeve A's contribution without touching B's."""
    close = flat_close()
    close["A/USD"] = 100.0
    close["B/USD"] = 100.0
    close.loc[close.index[14:], "A/USD"] = 50.0  # -50%: only A's own book draws down
    sleeve_a = np.zeros((len(close), 2))
    sleeve_a[:, 0] = 0.5  # sleeve A only ever holds A/USD
    sleeve_b = np.zeros((len(close), 2))
    sleeve_b[:, 1] = 0.5  # sleeve B only ever holds B/USD (flat price, never draws down)
    starts = window_starts(close.index, BT)
    breaker = BreakerConfig(dd_level=0.1, dd_mult=0.5, dd_recover=0.05)

    eq_multi, _ = simulate_multi_sleeve({"a": sleeve_a, "b": sleeve_b}, close, starts, BT, EX, breaker)
    eq_a_alone, _ = simulate(sleeve_a, close, starts, BT, EX, breaker)  # whole-book breaker on A alone
    eq_b_alone_no_brake, _ = simulate(sleeve_b, close, starts, BT, EX, NO_BRAKE)

    # B/USD never moves, so B's contribution to the combined book is identical whether or not the
    # (whole-book-equivalent) breaker exists -- the combined equity's final level, relative to
    # starting capital, reflects A's braked loss plus B's completely unaffected flat contribution.
    a_pnl = eq_a_alone[0, -1] - BT.capital
    b_pnl = eq_b_alone_no_brake[0, -1] - BT.capital
    assert abs(eq_multi[0, -1] - (BT.capital + a_pnl + b_pnl)) < 50.0  # a few dollars of fee/rounding slack
    assert abs(b_pnl - (-50.0)) < 1.0  # B/USD's price never moved; -$50 is just the one-time entry fee


def test_a_small_sleeve_is_scaled_to_its_own_typical_exposure():
    """A sleeve that only ever runs at 10% gross must still trip a 10%-drawdown breaker on a price
    move proportional to ITS OWN scale, not one calibrated for a ~100%-gross combined book -- a
    -50% move while the sleeve holds 10% notional is a real -50% loss on what that sleeve actually
    risks, not a harmless -5% dip against the $100k base."""
    close = flat_close()
    close["A/USD"] = 100.0
    close["B/USD"] = 100.0
    close.loc[close.index[14:], "A/USD"] = 50.0  # -50% crash
    small_sleeve = np.zeros((len(close), 2))
    small_sleeve[:, 0] = 0.1  # only 10% gross -- unscaled, a 10% dd_level would never fire
    other = np.zeros((len(close), 2))
    other[:, 1] = 0.5
    starts = window_starts(close.index, BT)
    breaker = BreakerConfig(dd_level=0.10, dd_mult=0.5, dd_recover=0.05)

    sleeves = {"small": small_sleeve, "other": other}
    eq_braked, _ = simulate_multi_sleeve(sleeves, close, starts, BT, EX, breaker)
    eq_unbraked, _ = simulate_multi_sleeve(sleeves, close, starts, BT, EX, NO_BRAKE)
    assert eq_braked[0, -1] > eq_unbraked[0, -1]  # the crashing small sleeve got dampened


def test_per_sleeve_beats_whole_book_when_only_one_sleeve_crashes():
    """The whole point: if only one sleeve is drawing down, a per-sleeve breaker should leave the
    healthy sleeve fully invested, so the combined book should end up ahead of a whole-book breaker
    that cuts everything (including the healthy sleeve) the moment the crashing one trips it."""
    close = flat_close(hours=24 * 5)
    close["A/USD"] = 100.0
    close["B/USD"] = np.linspace(100.0, 120.0, len(close))  # B/USD trends up the whole window
    close.loc[close.index[14:], "A/USD"] = 50.0  # A/USD crashes
    sleeve_a = np.zeros((len(close), 2))
    sleeve_a[:, 0] = 0.5
    sleeve_b = np.zeros((len(close), 2))
    sleeve_b[:, 1] = 0.5
    combined = sleeve_a + sleeve_b
    starts = window_starts(close.index, BT)
    breaker = BreakerConfig(dd_level=0.1, dd_mult=0.5, dd_recover=0.05)

    eq_whole, _ = simulate(combined, close, starts, BT, EX, breaker)
    eq_per_sleeve, _ = simulate_multi_sleeve(
        {"a": sleeve_a, "b": sleeve_b}, close, starts, BT, EX, breaker
    )
    assert eq_per_sleeve[0, -1] > eq_whole[0, -1]
