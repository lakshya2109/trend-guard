"""Crash hedge (book.crash_hedge): off by default, never part of the sleeves, shorts the instruments
only while BTC is falling fast, sized to the book's net long crypto, and never levers."""

import numpy as np
import pandas as pd
import pytest

from tests.test_book import CFG, synthetic_panel
from trend_guard.strategy.book import book_components, book_weights, crash_hedge

INST = ["COINB/USD", "MSTRB/USD"]


def panel_with_crash(crash_day: int = 70):
    """Synthetic panel plus BTC (a -15% drop over 12 hours on `crash_day`) and the two hedge tokens."""
    close, qv = synthetic_panel()
    idx = close.index
    rng = np.random.default_rng(1)
    wiggle = lambda: np.exp(rng.normal(0, 0.0005, len(idx)))  # tiny noise: real prices are never flat
    btc = np.full(len(idx), 60_000.0)
    t0 = crash_day * 24 + 3  # the drop starts at 03:00 UTC, between two check hours
    btc[t0 : t0 + 12] = 60_000 * np.linspace(1.0, 0.85, 12)
    btc[t0 + 12 :] = 51_000.0
    close["BTC/USD"] = btc * wiggle()
    for p in INST:
        close[p] = 300.0 * wiggle()
    qv = qv.reindex(columns=close.columns).fillna(1e7)
    return close, qv, t0


def enabled(**kw):
    h = CFG.book.crash_hedge.model_copy(update={"enabled": True, **kw})
    return CFG.book.model_copy(update={"crash_hedge": h})


def test_off_by_default_and_instruments_never_in_the_sleeves():
    assert CFG.book.crash_hedge.enabled is False
    close, qv, _ = panel_with_crash()
    W = book_weights(close, qv, CFG.book)
    assert (W[INST] == 0).all().all()  # the sleeves never touch the hedge tokens, even when listed
    base = book_weights(close.drop(columns=INST), qv.drop(columns=INST), CFG.book)
    np.testing.assert_allclose(W.drop(columns=INST).to_numpy(), base.to_numpy(), atol=1e-12)


def test_hedge_turns_on_at_the_next_check_hour_and_is_sized_to_net_long():
    close, qv, t0 = panel_with_crash()
    comps = book_components(close, qv, enabled())
    h = comps["crash_hedge"]
    idx = close.index
    cols = [close.columns.get_loc(p) for p in INST]
    active = (h[:, cols] < 0).any(axis=1)
    assert not active[:t0].any()  # quiet market: no hedge
    first = int(np.flatnonzero(active)[0])
    assert idx[first].hour in (0, 12) and first > t0  # switched on at a check hour, after the drop
    book = sum(v for k, v in comps.items() if k != "crash_hedge")
    net_long = max(book[first].sum(), 0.0)
    np.testing.assert_allclose(h[first, cols].sum(), -0.75 * net_long, rtol=1e-9)  # 75%, split evenly
    np.testing.assert_allclose(h[first, cols[0]], h[first, cols[1]])


def test_hedge_switches_off_once_btc_stops_falling():
    close, qv, t0 = panel_with_crash()
    h = book_components(close, qv, enabled())["crash_hedge"]
    cols = [close.columns.get_loc(p) for p in INST]
    active = (h[:, cols] < 0).any(axis=1)
    assert active.any() and not active[t0 + 24 * 5 :].any()  # BTC flat for days afterwards: hedge off


def test_hedge_never_levers_and_respects_short_switch():
    close, qv, _ = panel_with_crash()
    W = book_weights(close, qv, enabled(size=5.0)).to_numpy()  # absurd size: the gross cap must hold
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-9
    no_shorts = book_weights(close, qv, enabled(), allow_short=False)
    assert (no_shorts[INST] == 0).all().all()


def test_missing_instrument_prices_get_no_weight():
    close, qv, _ = panel_with_crash()
    close.loc[:, "MSTRB/USD"] = np.nan
    book = np.zeros(close.shape); book[:, 0] = 0.5
    h = crash_hedge(close, book, enabled().crash_hedge)
    assert (h[:, close.columns.get_loc("MSTRB/USD")] == 0).all()
    assert (h[:, close.columns.get_loc("COINB/USD")] <= 0).all() and (h[:, close.columns.get_loc("COINB/USD")] < 0).any()
