"""Rolling 14-day evaluation and the parameter-selection objective. BUILD_SPEC.md Section 11.3.

Every window starts at 00:00 UTC on each day (1-day step) and spans `window_days` full days. Daily
returns use UTC-close equity (the last hourly mark of each UTC day), matching the community rebuild
of the organizers' leaderboard; hourly returns use every hourly mark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_guard.metrics import all_definitions

STRESS_WINDOWS = {
    "2024-08 yen-carry unwind": ("2024-08-01", "2024-08-15"),
    "2025-02/03 drawdown": ("2025-02-20", "2025-03-06"),
    "2025-10 liquidation crash": ("2025-10-05", "2025-10-19"),
}
PERCENTILES = [10, 25, 50, 75, 90]


@dataclass
class Objective:
    """Section 11.3: maximise p25 daily_raw composite subject to three constraints."""

    min_median_return: float = 0.02
    max_median_red_share: float = 0.30
    max_p95_mdd: float = 0.05


def window_metrics(equity: pd.Series, trades: pd.DataFrame | None, start: pd.Timestamp, days: int) -> dict:
    """Metrics for one window [start, start + days). Equity is hourly, indexed by UTC bar time."""
    end = start + pd.Timedelta(days=days)
    base_slice = equity[equity.index < start]
    seg = equity[(equity.index >= start) & (equity.index < end)]
    if len(seg) < 24 * days * 0.9:
        return {}
    base = float(base_slice.iloc[-1]) if len(base_slice) else float(seg.iloc[0])
    daily_close = seg.resample("1D").last().dropna()
    daily = pd.concat([pd.Series([base]), daily_close.reset_index(drop=True)]).pct_change().dropna()
    hourly = pd.concat([pd.Series([base]), seg.reset_index(drop=True)]).pct_change().dropna()
    res = all_definitions(daily, hourly)
    row = {
        "start": start,
        "return": float(seg.iloc[-1] / base - 1.0),
        "mdd": res["daily_raw"].max_drawdown,
        "mdd_hourly": res["hourly_raw"].max_drawdown,
        "red_share": res["daily_raw"].red_share,
        "degenerate": any(r.denom_hit_eps for r in res.values()),
    }
    for name, r in res.items():
        row[f"composite_{name}"] = r.composite
        row[f"sharpe_{name}"] = r.sharpe
        row[f"sortino_{name}"] = r.sortino
        row[f"calmar_{name}"] = r.calmar
    if trades is not None and len(trades):
        tw = trades[(trades["ts"] >= start) & (trades["ts"] < end)]
        row["trades"] = len(tw)
        row["fees"] = float(tw["fee"].sum())
        row["turnover"] = float(tw["notional"].sum() / seg.mean())
        row["active_days"] = int(tw["ts"].dt.floor("D").nunique())
    else:
        row.update(trades=0, fees=0.0, turnover=0.0, active_days=0)
    return row


def rolling_windows(equity: pd.Series, trades: pd.DataFrame | None = None, days: int = 14) -> pd.DataFrame:
    first = equity.index[0].ceil("D")
    last_start = equity.index[-1].floor("D") - pd.Timedelta(days=days)
    rows = [window_metrics(equity, trades, s, days) for s in pd.date_range(first, last_start, freq="1D")]
    return pd.DataFrame([r for r in rows if r])


def distribution(windows: pd.DataFrame) -> pd.DataFrame:
    """p10/p25/median/p75/p90 of the headline columns."""
    cols = [
        "return",
        "composite_daily_raw",
        "composite_daily_ann",
        "composite_hourly_raw",
        "mdd",
        "red_share",
    ]
    cols = [c for c in cols if c in windows]
    return pd.DataFrame({f"p{q}": windows[cols].quantile(q / 100) for q in PERCENTILES}).rename(
        columns={"p50": "median"}
    )


def summarize(windows: pd.DataFrame) -> dict:
    if windows.empty:
        return {}
    return {
        "p25_composite_daily_raw": float(windows["composite_daily_raw"].quantile(0.25)),
        "median_composite_daily_raw": float(windows["composite_daily_raw"].median()),
        "p25_composite_daily_ann": float(windows["composite_daily_ann"].quantile(0.25)),
        "median_composite_daily_ann": float(windows["composite_daily_ann"].median()),
        "p25_composite_hourly_raw": float(windows["composite_hourly_raw"].quantile(0.25)),
        "median_composite_hourly_raw": float(windows["composite_hourly_raw"].median()),
        "median_return": float(windows["return"].median()),
        "p25_return": float(windows["return"].quantile(0.25)),
        "share_positive": float((windows["return"] > 0).mean()),
        "p95_mdd": float(windows["mdd"].quantile(0.95)),
        "median_red_share": float(windows["red_share"].median()),
        "median_trades": float(windows["trades"].median()),
        "median_fees": float(windows["fees"].median()),
        "median_active_days": float(windows["active_days"].median()),
        "n_windows": int(len(windows)),
    }


def passes(summary: dict, obj: Objective) -> dict[str, bool]:
    return {
        "median_return": summary.get("median_return", -1) >= obj.min_median_return,
        "median_red_share": summary.get("median_red_share", 1) <= obj.max_median_red_share,
        "p95_mdd": summary.get("p95_mdd", 1) <= obj.max_p95_mdd,
    }


def score(summary: dict, obj: Objective) -> float:
    """Selection score: p25 daily_raw composite if all constraints pass, else -inf-ish (ranked last,
    ordered by how close they came so the table is still informative)."""
    if not summary:
        return -np.inf
    ok = passes(summary, obj)
    base = summary["p25_composite_daily_raw"]
    return base if all(ok.values()) else base - 100.0 * (3 - sum(ok.values()))


def stress(equity: pd.Series, trades: pd.DataFrame | None, days: int = 14) -> pd.DataFrame:
    rows = []
    windows = dict(STRESS_WINDOWS)
    recent_start = (equity.index[-1].floor("D") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    windows["most recent 14 days"] = (recent_start, None)
    for name, (s, _) in windows.items():
        m = window_metrics(equity, trades, pd.Timestamp(s, tz="UTC"), days)
        if m:
            rows.append({"window": name, **m})
    return pd.DataFrame(rows)
