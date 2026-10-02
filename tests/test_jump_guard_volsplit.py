"""Jump guard (no new / larger longs after a pump) and the inverse-vol split inside momentum sides."""

import numpy as np
import pandas as pd

from trend_guard.strategy.book import _side_split, jump_guard


def panel(prices: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    return pd.DataFrame(prices, index=pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"))


def test_new_long_is_blocked_for_hold_hours_after_a_jump_then_allowed():
    close = panel({"A/USD": [100, 100, 115, 115, 115, 115, 115]})  # +15% in bar 2
    book = np.array([[0.0], [0.0], [0.0], [0.2], [0.2], [0.2], [0.2]])
    out = jump_guard(close, book, threshold=0.10, hold_h=2)
    assert list(out[:, 0]) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2]  # bars 3-4 inside the hold, 5 is free


def test_held_long_is_not_sold_or_added_to_and_reductions_pass():
    close = panel({"A/USD": [100, 100, 115, 115, 115]})
    book = np.array([[0.1], [0.1], [0.3], [0.05], [0.2]])
    out = jump_guard(close, book, threshold=0.10, hold_h=5)
    assert list(out[:, 0]) == [0.1, 0.1, 0.1, 0.05, 0.05]  # add blocked, cut allowed, re-add blocked


def test_shorts_and_small_moves_are_untouched():
    close = panel({"A/USD": [100, 100, 115, 115], "B/USD": [100, 105, 110, 115]})  # B: +5% bars only
    book = np.array([[0.0, 0.0], [0.0, 0.0], [-0.2, 0.3], [-0.3, 0.3]])
    out = jump_guard(close, book, threshold=0.10, hold_h=24)
    assert np.allclose(out, book)


def test_jump_guard_is_causal():
    rng = np.random.default_rng(0)
    close = panel(
        {
            "A/USD": list(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 200)))),
            "B/USD": list(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 200)))),
        }
    )
    book = np.abs(rng.normal(0.1, 0.1, (200, 2))) * (rng.random((200, 2)) > 0.3)
    full = jump_guard(close, book, 0.08, 12)
    cut = jump_guard(close.iloc[:120], book[:120], 0.08, 12)
    assert np.allclose(full[:120], cut)  # no target ever depends on a later bar


def test_side_split_equal_at_zero_and_favours_calm_names():
    vol = np.array([0.02, 0.08])
    assert np.allclose(_side_split(vol, 0.0), [0.5, 0.5])
    s = _side_split(vol, 0.5)
    assert np.isclose(s.sum(), 1.0) and s[0] > s[1]
    assert np.isclose(s[0] / s[1], 2.0)  # sqrt(0.08 / 0.02)
