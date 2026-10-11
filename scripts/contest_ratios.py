"""Live contest ratios and score from the bot's own equity record, the way scripts/simulate.py reports them.

Daily marks every 24 h from the contest start (backtest/session.py `metrics`), with the latest equity as the
last mark, so every number reads "as if the contest ended now". Daily returns, not annualised:
  Sortino = mean daily return / downside deviation     Sharpe = mean daily return / std
  Calmar  = mean daily return / max drawdown of the daily marks
  score   = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar
plus the previous edition's leaderboard Calmar (total return / max drawdown of the full record).
Read-only (the state db is opened with mode=ro). Stdlib only and Python 3.8+, so the log viewer can use
it with the box's system python3.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

EPS = 1e-8  # the simulator's floor for a zero denominator
DAY = timedelta(days=1)


def contest_start(config: Path) -> datetime:
    """competition.start_utc from config/strategy.yaml (a regex: PyYAML is not in the system python)."""
    m = re.search(r'^\s*start_utc:\s*"([^"]+)"', config.read_text(encoding="utf-8"), re.M)
    if not m:
        raise ValueError(f"no competition start_utc in {config}")
    return datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))


def equity_record(db: Path, since: datetime) -> list[tuple[datetime, float]]:
    """(time, equity) rows of the bot's per-minute equity table from `since` on, oldest first."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        rows = con.execute(
            "SELECT ts_utc, equity FROM equity WHERE ts_utc >= ? ORDER BY ts_utc", (since.isoformat()[:16],)
        ).fetchall()
    finally:
        con.close()
    return [(datetime.fromisoformat(ts), float(eq)) for ts, eq in rows]


def _max_dd(values: list[float]) -> float:
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = max(worst, 1.0 - v / peak)
    return worst


def ratios(points: list[tuple[datetime, float]], start: datetime, capital: float) -> dict | None:
    """Score and ratios from `capital` at `start` to the latest point; None before the first point."""
    if not points:
        return None
    now = points[-1][0]
    marks, last, i, t = [capital], capital, 0, start + DAY
    while t <= now:  # each full contest day: the last equity at or before the mark
        while i < len(points) and points[i][0] <= t:
            last = points[i][1]
            i += 1
        marks.append(last)
        t += DAY
    full_days = len(marks) - 1
    if now > t - DAY:  # the day in progress, as if the contest ended now
        marks.append(points[-1][1])
    r = [b / a - 1.0 for a, b in zip(marks, marks[1:])]  # noqa: B905 -- zip(strict) needs 3.10
    n = len(r)
    mean = sum(r) / n if n else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / n) if n else 0.0
    down = math.sqrt(sum(min(x, 0.0) ** 2 for x in r) / n) if n else 0.0
    mdd_daily = _max_dd(marks)
    sortino = mean / max(down, EPS)
    sharpe = mean / max(std, EPS)
    calmar = mean / max(mdd_daily, EPS)
    total = points[-1][1] / capital - 1.0
    mdd_full = _max_dd([capital] + [e for _, e in points])
    return {
        "score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
        "sortino": sortino,
        "sharpe": sharpe,
        "calmar": calmar,
        "calmar_total": total / max(mdd_full, EPS),
        "total_return": total,
        "equity": points[-1][1],
        "max_dd_daily": mdd_daily,
        "max_dd": mdd_full,
        "daily_returns": r,
        "full_days": full_days,
        "hours_into_day": (now - (start + full_days * DAY)).total_seconds() / 3600,
        "as_of": now,
    }


def _num(x: float) -> str:
    return f"{x:+.3f}" if abs(x) < 1e3 else ("+big" if x > 0 else "-big")  # no losing day yet: no downside


def summary_line(m: dict) -> str:
    """One line for the log viewer."""
    return (
        f"score {_num(m['score'])}  Sortino {_num(m['sortino'])}  Sharpe {_num(m['sharpe'])}  "
        f"Calmar {_num(m['calmar'])}  |  return {m['total_return']:+.2%}  max DD {m['max_dd']:.2%}  "
        f"Calmar(total/DD) {_num(m['calmar_total'])}  |  day {m['full_days'] + 1} "
        f"(+{m['hours_into_day']:.0f}h), as if it ended now"
    )


def report_lines(m: dict) -> list[str]:
    """A block for the portfolio view, laid out like scripts/simulate.py's CONTEST RATIOS."""
    days = "  ".join(f"{x:+.2%}" for x in m["daily_returns"])
    return [
        f"CONTEST RATIOS  (daily marks from the start, non-annualised, as if the contest ended "
        f"{m['as_of']:%d %b %H:%M} UTC)",
        f"  score = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar   {_num(m['score'])}",
        f"  Sortino   {_num(m['sortino'])}   (annualised ~{m['sortino'] * math.sqrt(365):+.2f})",
        f"  Sharpe    {_num(m['sharpe'])}   (annualised ~{m['sharpe'] * math.sqrt(365):+.2f})",
        f"  Calmar    {_num(m['calmar'])}   (mean daily return / max daily drawdown {m['max_dd_daily']:.2%})",
        f"  Calmar (total return / max drawdown)   {_num(m['calmar_total'])}   "
        f"({m['total_return']:+.2%} / {m['max_dd']:.2%}, previous edition's leaderboard)",
        f"  daily returns (day {m['full_days'] + 1} in progress): {days}",
    ]
