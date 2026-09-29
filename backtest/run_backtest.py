"""Backtest CLI. BUILD_SPEC.md Section 11.

python backtest/run_backtest.py --config config/strategy.yaml
    -> default config, long-only and no-overlay ablations, and the BTC / equal-weight / cash
       baselines, evaluated over every rolling 14-day window. Report under reports/<timestamp>/.

python backtest/run_backtest.py --config config/strategy.yaml --grid --workers 8
    -> also sweeps the Section 11.3 grid, ranks every config by the selection objective and
       reports PBO (CSCV, 16 partitions) for the selected config.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python backtest/run_backtest.py` from the repo root

import pandas as pd  # noqa: E402

from backtest.engine import (  # noqa: E402
    Variant,
    buy_and_hold,
    cash,
    equal_weight_hold,
    load_bars,
    run_backtest,
)
from backtest.pbo import pbo  # noqa: E402
from backtest.report import write_report  # noqa: E402
from backtest.windows import Objective, distribution, rolling_windows, score, stress, summarize  # noqa: E402
from trend_guard.config import StrategyConfig, config_hash, load_strategy_config, load_symbols  # noqa: E402

# Section 11.3 grid. The full 3x3x3x3 cross is 81 configs, above the spec's "under 50" limit, so
# deadband is swept only around the default of the other three: 27 + 2 = 29 configs.
GRID_SIGMA = [0.20, 0.30, 0.40]
GRID_LOOKBACKS = ["hourly", "daily", "blend"]
GRID_DD_MAX = [0.04, 0.05, 0.07]
GRID_DEADBAND_EXTRA = [0.015, 0.03]

_BARS: dict | None = None
_SYMBOLS = None


def grid_configs(base: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
    out = []

    def make(sigma: float, mode: str, dd: float, deadband: float) -> tuple[str, StrategyConfig]:
        c = copy.deepcopy(base)
        c.portfolio.sigma_portfolio_target = sigma
        c.signal.mode = mode  # type: ignore[assignment]
        c.risk.dd_max = dd
        c.execution.deadband = deadband
        return f"sig{sigma}_{mode}_dd{dd}_db{deadband}", c

    for sigma, mode, dd in product(GRID_SIGMA, GRID_LOOKBACKS, GRID_DD_MAX):
        out.append(make(sigma, mode, dd, base.execution.deadband))
    for db in GRID_DEADBAND_EXTRA:
        out.append(make(base.portfolio.sigma_portfolio_target, base.signal.mode, base.risk.dd_max, db))
    return out


def _init_worker(bars_dir: str) -> None:
    global _BARS, _SYMBOLS
    _SYMBOLS = load_symbols(ROOT / "config" / "symbols.yaml")
    _BARS = load_bars(Path(bars_dir), _SYMBOLS)


def _run_one(args: tuple[str, StrategyConfig, str, str | None]) -> tuple[str, pd.Series, pd.DataFrame, dict]:
    name, cfg, start, end = args
    res = run_backtest(_BARS, cfg, _SYMBOLS, start, end)
    return name, res.equity, res.trades, res.meta


def _daily_returns(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().pct_change().dropna()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/strategy.yaml")
    ap.add_argument("--bars", default="data/bars/1h")
    ap.add_argument("--start", default=None, help="default: backtest.start from the config")
    ap.add_argument("--end", default=None)
    ap.add_argument("--grid", action="store_true", help="run the Section 11.3 parameter sweep + PBO")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    cfg = load_strategy_config(args.config)
    symbols = load_symbols(ROOT / "config" / "symbols.yaml")
    bars = load_bars(Path(args.bars), symbols)
    if not bars:
        print(f"no bars found in {args.bars}; run scripts/download_history.py first", file=sys.stderr)
        return 1
    start = args.start or cfg.backtest.start
    days = cfg.backtest.window_days
    obj = Objective()
    t0 = time.time()
    print(f"loaded {len(bars)} pairs; backtesting from {start}")

    # --- default config + ablations -------------------------------------------------------------
    runs = {
        "strategy": Variant("strategy"),
        "long_only": Variant("long_only", long_only=True),
        "no_overlays": Variant("no_overlays", overlays=False),
    }
    equities: dict[str, pd.Series] = {}
    trades: dict[str, pd.DataFrame] = {}
    metas: dict[str, dict] = {}
    for name, v in runs.items():
        res = run_backtest(bars, cfg, symbols, start, args.end, v)
        equities[name], trades[name], metas[name] = res.equity, res.trades, res.meta
        print(
            f"  {name}: final equity {res.equity.iloc[-1]:,.0f}, {len(res.trades)} fills, "
            f"fees ${res.fees:,.0f} ({time.time() - t0:.0f}s)"
        )

    idx = equities["strategy"].index
    equities["btc_hold"] = buy_and_hold(bars, "BTC/USD", start, args.end).reindex(idx).ffill()
    equities["ew_top10_hold"] = equal_weight_hold(bars, symbols, start, 10, args.end).reindex(idx).ffill()
    equities["cash"] = cash(idx)

    windows = {n: rolling_windows(eq, trades.get(n), days) for n, eq in equities.items()}
    summaries = {n: summarize(w) for n, w in windows.items()}
    dists = {"strategy": distribution(windows["strategy"])}
    stresses = {n: stress(equities[n], trades.get(n), days) for n in ("strategy", "btc_hold")}

    extra: list[str] = [
        "## Run notes",
        "",
        f"- min free cash seen (strategy): ${metas['strategy']['min_cash']:,.2f} "
        "(negative would mean a taker fallback overspent the planned budget)",
        "- total fills: " + ", ".join(f"{n} {m['n_trades']}" for n, m in metas.items()),
        "",
    ]

    # --- grid + PBO -----------------------------------------------------------------------------
    if args.grid:
        configs = grid_configs(cfg)
        print(f"running grid of {len(configs)} configs on {args.workers} workers ...")
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.bars,)) as ex:
            results = list(ex.map(_run_one, [(n, c, start, args.end) for n, c in configs]))
        rows, daily = [], {}
        for name, eq, tr, _meta in results:
            s = summarize(rolling_windows(eq, tr, days))
            rows.append({"config": name, "score": score(s, obj), **s})
            daily[name] = _daily_returns(eq)
        grid_df = pd.DataFrame(rows).sort_values("score", ascending=False)
        best = grid_df.iloc[0]["config"]
        p = pbo(pd.DataFrame(daily), n_partitions=16)
        cols = [
            "config",
            "score",
            "p25_composite_daily_raw",
            "median_composite_daily_raw",
            "median_return",
            "p95_mdd",
            "median_red_share",
            "median_trades",
        ]
        extra += [
            "## Parameter grid (Section 11.3)",
            "",
            f"Selected by objective: **{best}**. PBO (CSCV, 16 partitions, {p['n_splits']} splits): "
            f"**{p['pbo']:.2%}** -- the probability that the in-sample winner ranks below the median "
            "out of sample.",
            "",
            grid_df[cols].to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
        print(f"grid done: best={best}, PBO={p['pbo']:.2%} ({time.time() - t0:.0f}s)")

    first, last = idx[0], idx[-1]
    data_desc = (
        f"{len(bars)} Binance USDT pairs as Roostoo /USD, 1h bars, "
        f"evaluated {first:%Y-%m-%d} to {last:%Y-%m-%d}"
    )
    out = write_report(
        Path(args.out),
        config_hash(args.config),
        data_desc,
        summaries,
        dists,
        stresses,
        windows,
        equities,
        extra,
        obj,
    )
    print(f"report written to {out / 'summary.md'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
