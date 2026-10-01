"""backtest/contest.py: hand-checkable simulator cases, the score, and the breaker."""

import numpy as np
import pandas as pd

from backtest.contest import contest_score, simulate, summarize, window_starts
from trend_guard.config import BacktestConfig, BreakerConfig, ExecutionConfig
from trend_guard.risk.breaker import Breaker

BT = BacktestConfig(
    first_window="2025-01-01", last_window="2025-01-01", window_days=2, fee=0.001, slippage=0.0
)
EX = ExecutionConfig()
NO_BRAKE = BreakerConfig(dd_level=1.0)


def flat_close(hours: int = 24 * 5, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame({"A/USD": price}, index=idx)


def test_window_starts_at_contest_hour():
    close = flat_close()
    starts = window_starts(close.index, BT)
    assert len(starts) == 1 and close.index[starts[0]].hour == BT.window_hour_utc


def test_buying_half_the_book_costs_exactly_the_fee():
    close = flat_close()
    W = np.full((len(close), 1), 0.5)
    eq, turn = simulate(W, close, window_starts(close.index, BT), BT, EX, NO_BRAKE)
    assert np.isclose(eq[0, -1], 100_000 - 50_000 * 0.001)  # one buy, then the target never changes
    assert np.isclose(turn[0], 0.5)


def test_price_move_and_short_pnl():
    close = flat_close()
    close.iloc[20:] = 110.0  # +10% after the window opens at hour 12
    for w, sign in ((0.5, 1), (-0.5, -1)):
        W = np.full((len(close), 1), w)
        eq, _ = simulate(W, close, window_starts(close.index, BT), BT, EX, NO_BRAKE)
        pnl = eq[0, -1] - 100_000
        assert np.sign(pnl) == sign
        assert abs(abs(pnl) - 5_000) < 250  # 10% of 50k, less fees and small drift rebalances


def test_breaker_cuts_exposure_after_a_crash():
    close = flat_close()
    close.iloc[14:] = 50.0  # -50% two hours into the window
    close.iloc[20:] = 100.0  # full recovery later
    W = np.full((len(close), 1), 1.0)
    starts = window_starts(close.index, BT)
    eq_no, _ = simulate(W, close, starts, BT, EX, NO_BRAKE)
    eq_br, _ = simulate(W, close, starts, BT, EX, BreakerConfig(dd_level=0.1, dd_mult=0.5, dd_recover=0.05))
    assert eq_br[0, -1] < eq_no[0, -1]  # sold half at the bottom: the breaker is a cost in V-shapes


def test_score_is_positive_for_steady_gains_and_ranks_smoothness():
    steady = np.cumprod(np.full(14 * 24 + 1, 1.0 + 0.002 / 24))[None, :] * 100_000
    rng = np.random.default_rng(0)
    noisy = np.cumprod(1.0 + 0.002 / 24 + rng.normal(0, 0.004, 14 * 24 + 1))[None, :] * 100_000
    s = contest_score(np.vstack([steady, noisy]))
    assert s[0] > 0 and s[0] > s[1]


def test_summarize_reports_the_team_columns():
    eq = np.array([[100.0] * 24 * 14 + [101.0], [100.0] * 24 * 14 + [98.0]])
    st = pd.DatetimeIndex(["2025-12-31 12:00", "2026-01-02 12:00"], tz="UTC")
    df = summarize(eq, np.zeros(2), st, {"2026": ("2026-01-01", None), "all": (None, None)})
    assert df.loc["all", "positive"] == 0.5 and np.isclose(df.loc["2026", "worst"], -0.02)
    assert df.loc["all", "above_-1%"] == 0.5


def test_breaker_state_machine():
    b = Breaker(BreakerConfig(dd_level=0.10, dd_mult=0.5, dd_recover=0.05))
    assert b.update(100.0) == 1.0
    assert b.update(89.0) == 0.5  # 11% below peak
    assert b.update(93.0) == 0.5  # 7%: still braked (hysteresis)
    assert b.update(96.0) == 1.0  # 4%: released
    b2 = Breaker(BreakerConfig())
    b2.load(b.state())
    assert b2.peak == 100.0


def test_competition_metrics_and_active_days():
    from backtest.contest import competition_metrics, non_overlapping

    close = flat_close()
    close.iloc[30:] = 110.0  # +10% inside the window
    W = np.full((len(close), 1), 0.5)
    starts = window_starts(close.index, BT)
    eq, _, traded = simulate(W, close, starts, BT, EX, NO_BRAKE, with_trades=True)
    m = competition_metrics(eq, traded, close.index[starts])
    row = m.iloc[0]
    assert row["active_days"] == 1  # one buy at the open, nothing after
    assert np.isclose(row["calmar_total"], row["return"] / row["max_drawdown"])
    assert list(non_overlapping(pd.date_range("2025-01-01", periods=30, freq="D"), 14)) == [0, 14, 28]


def test_rank_vs_previous_field():
    from backtest.benchmark import FIELD, WINNER, field_scores, rank_vs_field

    # the percentile normalisation orders last year's teams like their published scores
    s = field_scores(FIELD[:, 0], FIELD[:, 1], FIELD[:, 2], FIELD[:, 3])
    assert pd.Series(s).corr(pd.Series(FIELD[:, 4]), method="spearman") > 0.95
    assert rank_vs_field(WINNER["sharpe"] + 0.1, WINNER["sortino"] + 0.1, WINNER["calmar_total"] + 1, 15) == 1
    assert rank_vs_field(-2.0, -2.0, -10.0, 1) == len(FIELD) + 1
