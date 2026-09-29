"""Markdown + CSV report writer for backtest runs. BUILD_SPEC.md Section 11.3."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backtest.windows import Objective, passes


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt_pct(x: float) -> str:
    return f"{x:.2%}" if x == x else "n/a"


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x == x else "n/a"


def summary_table(summaries: dict[str, dict], obj: Objective) -> str:
    """One row per strategy/baseline with the headline numbers and constraint checks."""
    head = (
        "| run | p25 comp (daily_raw) | median comp (daily_raw) | p25 comp (daily_ann) | "
        "p25 comp (hourly_raw) | median 14d return | share windows > 0 | p95 MDD | median red days | "
        "median trades/14d | median fees/14d | checks |"
    )
    lines = [head, "|" + "---|" * 12]
    for name, s in summaries.items():
        if not s:
            continue
        ok = passes(s, obj)
        checks = "PASS" if all(ok.values()) else "FAIL: " + ", ".join(k for k, v in ok.items() if not v)
        lines.append(
            f"| {name} | {_fmt(s['p25_composite_daily_raw'])} | {_fmt(s['median_composite_daily_raw'])} | "
            f"{_fmt(s['p25_composite_daily_ann'])} | {_fmt(s['p25_composite_hourly_raw'])} | "
            f"{_fmt_pct(s['median_return'])} | {_fmt_pct(s['share_positive'])} | {_fmt_pct(s['p95_mdd'])} | "
            f"{_fmt_pct(s['median_red_share'])} | {s['median_trades']:.0f} | ${s['median_fees']:,.0f} | "
            f"{checks} |"
        )
    return "\n".join(lines)


def write_report(
    out_root: Path,
    cfg_hash: str,
    data_desc: str,
    summaries: dict[str, dict],
    distributions: dict[str, pd.DataFrame],
    stress_tables: dict[str, pd.DataFrame],
    windows: dict[str, pd.DataFrame],
    equities: dict[str, pd.Series],
    extra_sections: list[str] | None = None,
    obj: Objective | None = None,
) -> Path:
    obj = obj or Objective()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / stamp
    out.mkdir(parents=True, exist_ok=True)

    md = [
        "# Backtest report",
        "",
        f"- generated: {stamp}  commit: `{git_commit()}`  config hash: `{cfg_hash}`",
        f"- data: {data_desc}",
        f"- objective (Section 11.3): maximise p25 daily_raw composite s.t. median 14d return >= "
        f"{obj.min_median_return:.0%}, median red-day share <= {obj.max_median_red_share:.0%}, "
        f"p95 14d MDD <= {obj.max_p95_mdd:.0%}",
        "- composite = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar; windows = every 14-day window, 1-day step",
        "",
        "## Strategy vs baselines",
        "",
        summary_table(summaries, obj),
        "",
    ]
    for name, dist in distributions.items():
        md += [f"## Distribution across 14-day windows: {name}", "", dist.to_markdown(floatfmt=".4f"), ""]
    for name, st in stress_tables.items():
        if st.empty:
            continue
        cols = ["window", "start", "return", "mdd", "red_share", "composite_daily_raw", "trades", "fees"]
        md += [
            f"## Stress windows: {name}",
            "",
            st[[c for c in cols if c in st]].to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    md += extra_sections or []
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")

    for name, w in windows.items():
        w.to_csv(out / f"windows_{name}.csv", index=False)
    pd.DataFrame(equities).to_csv(out / "equity_hourly.csv")
    pd.DataFrame(summaries).T.to_csv(out / "summary.csv")
    return out
