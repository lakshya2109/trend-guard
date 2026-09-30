"""Simulated session: matches the contest simulator, logs consistent trades, and detects no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.contest import contest_score, simulate
from backtest.session import _classify, metrics, run_session
from trend_guard.config import load_strategy_config
from trend_guard.strategy.book import book_weights


def _panel(T: int = 24 * 70, N: int = 8, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=T, freq="1h", tz="UTC")
    drift = np.linspace(-4e-4, 4e-4, N)  # some coins trend up, some down
    ret = rng.normal(drift, 0.01, size=(T, N))
    cols = [f"C{i}/USD" for i in range(N)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    qv = pd.DataFrame(5e8, index=idx, columns=cols)
    return close, qv


def _cfg():
    cfg = load_strategy_config("config/strategy.yaml")
    book = cfg.book.model_copy(update={"team_share": 0.0, "v2_share": 1.0})  # keep the test fast
    book = book.model_copy(
        update={
            "trend": book.trend.model_copy(update={"min_age_h": 48, "min_liquidity_usd": 0}),
            "momentum": book.momentum.model_copy(update={"min_age_h": 48, "min_liquidity_usd": 0, "k": 2}),
        }
    )
    return cfg.model_copy(update={"book": book})


def test_session_matches_contest_simulator() -> None:
    close, qv = _panel()
    cfg = _cfg()
    start = close.index[24 * 50 + 12]
    end = start + pd.Timedelta(days=14)
    c, q = close[close.index <= end], qv[qv.index <= end]
    res = run_session(c, q, cfg, start, end)
    W = book_weights(c, q, cfg.book).to_numpy()
    eq, _ = simulate(W, c, np.array([c.index.get_loc(start)]), cfg.backtest, cfg.execution, cfg.breaker)
    assert np.allclose(res.equity.to_numpy(), eq[0])
    assert res.leakage_ok
    assert len(res.trades) > 0
    m = metrics(res, cfg.backtest.capital)
    assert np.isclose(m["score"], contest_score(eq)[0])
    # fees in the log add up to the cash spent on costs
    cost = cfg.backtest.fee + cfg.backtest.slippage
    assert np.isclose(m["fees_paid"], sum(t.notional for t in res.trades) * cost)


def test_session_refuses_future_bars() -> None:
    close, qv = _panel()
    start = close.index[24 * 50]
    try:
        run_session(close, qv, _cfg(), start, start + pd.Timedelta(days=3))
    except AssertionError:
        return
    raise AssertionError("a panel running past the session end must be rejected")


def test_classify_flip_is_two_legs() -> None:
    assert _classify(0.1, -0.05) == [("SELL", "flip", 0.1, 0.0), ("SHORT", "flip", 0.0, -0.05)]
    assert _classify(0.0, 0.02)[0][:2] == ("BUY", "open")
    assert _classify(-0.03, 0.0)[0][:2] == ("COVER", "close")
    assert _classify(-0.03, -0.05)[0][:2] == ("SHORT", "add")
