"""Net hedge: a short basket of the most liquid coins against part of the book's net exposure."""

import numpy as np
import pandas as pd

from tests.test_book import CFG, synthetic_panel
from trend_guard.config import NetHedgeConfig
from trend_guard.strategy.book import book_components, book_weights, net_hedge

ON = CFG.book.model_copy(update={"net_hedge": NetHedgeConfig(enabled=True, frac=0.5, n_coins=4)})
OFF = CFG.book.model_copy(update={"net_hedge": NetHedgeConfig(enabled=False)})


def test_basket_is_the_most_liquid_old_enough_coins_and_never_paxg():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="1h", tz="UTC")
    close = pd.DataFrame(100.0, index=idx, columns=["A/USD", "B/USD", "C/USD", "NEW/USD", "PAXG/USD"])
    close.loc[idx[: 24 * 20], "NEW/USD"] = np.nan  # listed on day 20: too young for 30 days
    qv = pd.DataFrame({"A/USD": 3e8, "B/USD": 2e8, "C/USD": 1e3, "NEW/USD": 9e8, "PAXG/USD": 9e9}, index=idx)
    book = np.zeros(close.shape)
    book[:, 0] = 0.6  # net +0.6
    h = net_hedge(close, qv, book, NetHedgeConfig(frac=0.5, n_coins=3, min_liquidity_usd=1e6))
    # C is illiquid, NEW too young, PAXG excluded: A and B share the 0.3 hedge
    np.testing.assert_allclose(h[-1], [-0.15, -0.15, 0.0, 0.0, 0.0])
    assert (h[: 24 * 7] == 0).all()  # no trailing 7-day liquidity yet: no basket, no hedge


def test_hedge_halves_the_net_without_leverage_and_off_means_nothing_changes():
    close, qv = synthetic_panel()
    off = book_components(close, qv, OFF)
    assert not off["net_hedge"].any()
    W0 = sum(off.values())
    W = book_weights(close, qv, ON).to_numpy()
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-9
    net0, net = W0.sum(axis=1), W.sum(axis=1)
    big = np.abs(net0) > 0.1
    assert big.any()
    assert (np.abs(net[big]) < np.abs(net0[big])).all()  # every hour with real exposure is hedged
    room = big & (np.abs(W0).sum(axis=1) + 0.5 * np.abs(net0) <= 1.0 - 1e-6)  # gross cap not binding
    np.testing.assert_allclose(net[room], 0.5 * net0[room], atol=1e-9)


def test_hedge_is_causal():
    close, qv = synthetic_panel()
    full = book_weights(close, qv, ON)
    cut = book_weights(close.iloc[:-200], qv.iloc[:-200], ON)
    np.testing.assert_allclose(full.iloc[:-200].to_numpy(), cut.to_numpy(), atol=1e-12)
