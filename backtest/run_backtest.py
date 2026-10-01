"""Backtest CLI: replay every 14-day contest window for the book, each sleeve alone, and baselines.

    python scripts/download_history.py                          # fill data/bars/1h first
    python backtest/run_backtest.py --config config/strategy.yaml

Writes reports/<timestamp>/summary.md and windows.csv, and prints the comparison table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python backtest/run_backtest.py` from the repo root
sys.path.insert(0, str(ROOT / "src"))  # import trend_guard even without `pip install -e .`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.benchmark import rank_vs_field  # noqa: E402
from backtest.contest import (  # noqa: E402
    competition_metrics,
    contest_score,
    non_overlapping,
    simulate,
    summarize,
    window_starts,
)
from trend_guard.config import config_hash, load_strategy_config, load_symbols  # noqa: E402
from trend_guard.data.binance_live import load_cache, panel  # noqa: E402
from trend_guard.strategy.book import book_weights  # noqa: E402

PERIODS = {"2024-25": (None, "2026-01-01"), "2026": ("2026-01-01", None), "all": (None, None)}
# The team's published combined book (roostoo-last-supper README, same windows, costs and score),
# shown for reference only: their code is not part of this repo.
REFERENCE = {
    "2026": {"score": 0.131, "median_return": 0.0091, "above_-1%": 0.84, "positive": 0.69, "worst": -0.022},
    "all": {"score": 0.059, "median_return": 0.0046, "above_-1%": 0.75, "positive": 0.59, "worst": -0.048},
}


def load_panel(
    bars_dir: Path, excluded: list[str], overrides: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inverse = {v: k for k, v in overrides.items()}
    symbols = sorted(p.stem for p in bars_dir.glob("*.parquet"))
    pair_of = {s: inverse.get(s, s[:-4] + "/USD") for s in symbols if s.endswith("USDT")}
    pair_of = {s: p for s, p in pair_of.items() if p not in excluded}
    return panel(load_cache(bars_dir, list(pair_of)), pair_of)


def baselines(close: pd.DataFrame, qv: pd.DataFrame) -> dict[str, np.ndarray]:
    """BTC buy-and-hold and an equal-weight top-10-by-liquidity book re-picked daily at 12:00."""
    T, N = close.shape
    btc = np.zeros((T, N))
    if "BTC/USD" in close:
        btc[:, close.columns.get_loc("BTC/USD")] = np.where(close["BTC/USD"].notna(), 1.0, 0.0)
    liq = (qv.rolling(7 * 24, min_periods=72).sum() / 7).to_numpy()
    aged = (close.notna().cumsum() >= 720).to_numpy()
    ew = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        if close.index[t].hour == 12:
            ok = aged[t] & np.isfinite(liq[t]) & (close.columns != "PAXG/USD")
            top = np.argsort(-np.where(ok, liq[t], -np.inf))[:10]
            cur = np.zeros(N)
            cur[[i for i in top if ok[i]]] = 0.1
        ew[t] = cur
    return {"btc_hold": btc, "ew_top10": ew, "cash": np.zeros((T, N))}


COMP_RUNS = ("book", "v2_only", "team_only", "btc_hold")


def competition_table(m: pd.DataFrame, days: int) -> pd.Series:
    """One run's non-overlapping fortnights, scored like the previous edition's leaderboard."""
    rank = np.array(
        [rank_vs_field(r.sharpe, r.sortino, r.calmar_total, r.active_days) for r in m.itertuples()]
    )
    sel = m.iloc[non_overlapping(m.index, days)]
    rk = rank[non_overlapping(m.index, days)]
    # how much the answer depends on which day the series starts: median return for each start offset
    offs = [m["return"].iloc[non_overlapping(m.index, days, o)].median() for o in range(min(days, len(m)))]
    return pd.Series(
        {
            "fortnights": len(sel),
            "median_return": sel["return"].median(),
            "mean_return": sel["return"].mean(),
            "P(>+5%)": (sel["return"] > 0.05).mean(),
            "P(<-5%)": (sel["return"] < -0.05).mean(),
            "positive": (sel["return"] > 0).mean(),
            "sharpe": sel["sharpe"].median(),
            "sortino": sel["sortino"].median(),
            "calmar_total": sel["calmar_total"].median(),
            "max_drawdown": sel["max_drawdown"].median(),
            "min_active_days": sel["active_days"].min(),
            "median_rank": np.median(rk),
            "P(rank 1)": (rk <= 1).mean(),
            "P(top 3)": (rk <= 3).mean(),
            "P(top 10)": (rk <= 10).mean(),
            "median_return_offset_lo": min(offs),
            "median_return_offset_hi": max(offs),
        }
    )


def competition_view(comp: dict[str, pd.DataFrame], days: int) -> list[str]:
    """Markdown + console tables: non-overlapping fortnights per year, previous edition's scoring."""
    md = [
        "## Competition view: non-overlapping fortnights",
        "",
        "Windows share no day (every 14th daily start). Ratios as the previous edition's leaderboard "
        "reported them: daily Sharpe and Sortino (not annualised), Calmar = total return / max drawdown, "
        "active days = dates with a trade. Rank = estimated finish among that edition's 52 teams "
        "(backtest/benchmark.py). `median_return_offset_lo/hi` = range of the median over the 14 possible "
        "start days.",
        "",
    ]
    for year in ("2024", "2025", "2026", "all"):
        rows = {}
        for name in COMP_RUNS:
            m = comp[name]
            if year != "all":
                m = m[m.index.year == int(year)]
            if len(m):
                rows[name] = competition_table(m, days)
        df = pd.DataFrame(rows).T
        md += [f"### {year}", "", df.to_markdown(floatfmt=".4f"), ""]
        print(f"\n=== competition view {year} (non-overlapping)\n{df.round(4).to_string()}")
    return md


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/strategy.yaml")
    ap.add_argument("--bars", default="data/bars/1h")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_strategy_config(args.config)
    sym = load_symbols(ROOT / "config" / "symbols.yaml")
    close, qv = load_panel(Path(args.bars), sym.excluded, sym.overrides)
    if close.empty:
        print(f"no bars in {args.bars}: run scripts/download_history.py first", file=sys.stderr)
        return 1
    b = cfg.book
    # "book" is exactly what the live engine trades; the other two show each part on its own
    runs = {
        "book": book_weights(close, qv, b, b.allow_short).to_numpy(),
        "v2_only": book_weights(
            close, qv, b.model_copy(update={"v2_share": 1.0, "team_share": 0.0})
        ).to_numpy(),
        "team_only": book_weights(
            close, qv, b.model_copy(update={"v2_share": 0.0, "team_share": 1.0})
        ).to_numpy(),
    }
    runs.update(baselines(close, qv))
    print(f"{close.shape[1]} pairs, {len(close)} hours; weights built in {time.time() - t0:.0f}s")

    starts = window_starts(close.index, cfg.backtest)
    st = close.index[starts]
    tables, windows, comp = {}, [], {}
    for name, W in runs.items():
        eq, turn, traded = simulate(
            W, close, starts, cfg.backtest, cfg.execution, cfg.breaker, with_trades=True
        )
        tables[name] = summarize(eq, turn, st, PERIODS)
        comp[name] = competition_metrics(eq, traded, st)
        windows.append(
            pd.DataFrame(
                {
                    "run": name,
                    "start": st,
                    "return": eq[:, -1] / eq[:, 0] - 1,
                    "score": contest_score(eq),
                    "turnover": turn,
                }
            )
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "score",
        "median_return",
        "above_-1%",
        "positive",
        "worst",
        "p10_return",
        "p95_drawdown",
        "turnover",
    ]
    md = [
        f"# Contest replay ({stamp})",
        "",
        f"commit `{git_commit()}`, config `{config_hash(args.config)}`, {close.shape[1]} pairs, "
        f"{len(starts)} windows of {cfg.backtest.window_days} days starting daily at "
        f"{cfg.backtest.window_hour_utc:02d}:00 UTC, "
        f"costs {cfg.backtest.fee + cfg.backtest.slippage:.2%} per side.",
        "",
    ]
    for period in PERIODS:
        rows = {n: t.loc[period] for n, t in tables.items() if period in t.index}
        df = pd.DataFrame(rows).T[cols]
        if period in REFERENCE:
            df.loc["team combined 70/30 (published)"] = pd.Series(REFERENCE[period])
        md += [
            f"## {period} ({int(tables['book'].loc[period, 'windows'])} windows)",
            "",
            df.to_markdown(floatfmt=".4f"),
            "",
        ]
        print(f"\n--- {period}\n{df.round(4).to_string()}")
    md += competition_view(comp, cfg.backtest.window_days)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    pd.concat(windows).to_csv(out / "windows.csv", index=False)
    print(f"\nreport: {out / 'summary.md'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
