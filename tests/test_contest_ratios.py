"""scripts/contest_ratios.py: the live score matches backtest/session.py's on the same equity curve."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.session import SessionResult, metrics
from scripts import contest_ratios as cr
from trend_guard.clock import SimClock
from trend_guard.state.store import Store

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 4, 12, tzinfo=UTC)


def curve(hours: int = 3 * 24 + 5) -> pd.Series:
    """Hourly equity from $100k at the start: a dip, a rally and a partial last day."""
    rng = np.random.default_rng(3)
    steps = 1 + rng.normal(0.0, 0.004, hours)
    eq = 100_000 * np.cumprod(np.r_[1.0, steps])
    return pd.Series(eq, index=pd.date_range(START, periods=hours + 1, freq="1h"))


def test_matches_the_simulator_report_including_the_day_in_progress():
    eq = curve()
    sim = metrics(
        SessionResult(eq, [], pd.DataFrame({"gross": [0.0], "net": [0.0]}), pd.Series(dtype=float), True, ""),
        100_000,
    )
    live = cr.ratios([(t.to_pydatetime(), float(v)) for t, v in eq.iloc[1:].items()], START, 100_000)
    assert live is not None
    for k in ("score", "sortino", "sharpe", "calmar", "calmar_total", "total_return"):
        assert abs(live[k] - sim[k]) < 1e-9, k
    assert live["full_days"] == 3 and abs(live["hours_into_day"] - 5) < 1e-9
    assert len(live["daily_returns"]) == 4  # three full days + the one in progress


def test_reads_the_bot_state_db_read_only_and_the_contest_start(tmp_path):
    store = Store(tmp_path / "live.db", SimClock(START))
    for k, e in enumerate((100_000, 99_000, 101_500)):
        ts = (START + timedelta(hours=12 * (k + 1))).isoformat()
        store.record_equity(ts, Decimal(e), Decimal(0), Decimal(0), Decimal(0), Decimal(101_500), Decimal(0))
    store.conn.commit()
    pts = cr.equity_record(tmp_path / "live.db", START)
    assert [e for _, e in pts] == [100_000, 99_000, 101_500]
    m = cr.ratios(pts, START, 100_000)
    assert m is not None and abs(m["total_return"] - 0.015) < 1e-12 and abs(m["max_dd"] - 0.01) < 1e-12
    assert "score" in cr.summary_line(m) and cr.report_lines(m)[0].startswith("CONTEST RATIOS")
    assert cr.contest_start(ROOT / "config/strategy.yaml") == START
    assert cr.ratios([], START, 100_000) is None
