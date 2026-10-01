"""Reproduce the design search behind config/strategy.yaml, and its probability of overfitting.

    python backtest/sweep.py

A pre-declared grid of 36 variants (< 50, per BUILD_SPEC Section 11.3):
  trend dust floor   min_weight 0.005 | 0.01 | both (ensemble)
  momentum members   lookbacks mid (3-14d, 7-30d) | all (+1-7d) x skips {12} | {0, 12}
  capital split      0.7/0.3 | 0.6/0.4 | 0.8/0.4 (trend / momentum)
Each is replayed on every 14-day contest window. Selection used the 2024-25 windows; 2026 was held out.
PBO is computed with CSCV (16 partitions) on the daily returns of one continuous run per variant.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # import trend_guard even without `pip install -e .`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.contest import _simulate, simulate, summarize, window_starts  # noqa: E402
from backtest.pbo import pbo  # noqa: E402
from backtest.run_backtest import PERIODS, load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import momentum_sleeve, trend_sleeve  # noqa: E402

DUST = {"mw0.005": [0.005], "mw0.01": [0.01], "mw_ens": [0.005, 0.01]}
LOOKBACKS = {
    "mid": [[72, 168, 336], [168, 336, 720]],
    "all": [[24, 72, 168], [72, 168, 336], [168, 336, 720]],
}
SKIPS = {"s12": [12], "s0_12": [0, 12]}
SPLITS = [(0.7, 0.3), (0.6, 0.4), (0.8, 0.4)]


def continuous_daily_returns(W: np.ndarray, close: pd.DataFrame, cfg) -> pd.Series:
    """One run from the first window to the end of data, daily marks at the window hour."""
    idx = close.index
    s = int(window_starts(idx, cfg.backtest)[0])
    px = close.ffill(limit=3).to_numpy(dtype=np.float64)
    act = np.isin(idx.hour, cfg.execution.decision_hours_utc)
    eq, _, _ = _simulate(
        np.ascontiguousarray(W, dtype=np.float64),
        px,
        act,
        np.array([s], dtype=np.int64),
        len(idx) - 1 - s,
        cfg.backtest.fee + cfg.backtest.slippage,
        cfg.execution.min_trade_weight,
        cfg.execution.min_trade_frac,
        cfg.backtest.capital,
        1.0,
        1.0,
        0.0,
    )
    marks = pd.Series(eq[0], index=idx[s:]).iloc[::24]
    return marks.pct_change().dropna()


def main() -> int:
    t0 = time.time()
    cfg = load_strategy_config(ROOT / "config" / "strategy.yaml")
    sym = load_symbols(ROOT / "config" / "symbols.yaml")
    close, qv = load_panel(ROOT / "data" / "bars" / "1h", sym.excluded, sym.overrides)
    b = cfg.book
    trends = {k: trend_sleeve(close, qv, b.trend, min_weights=v) for k, v in DUST.items()}
    moms = {
        f"{lk}|{sk}": momentum_sleeve(
            close, qv, b.momentum.model_copy(update={"lookback_sets": lv, "skips": sv})
        )
        for (lk, lv), (sk, sv) in itertools.product(LOOKBACKS.items(), SKIPS.items())
    }
    starts = window_starts(close.index, cfg.backtest)
    st = close.index[starts]
    rows, daily = [], {}
    for (tk, TR), (mk, MO), (a, m) in itertools.product(trends.items(), moms.items(), SPLITS):
        name = f"{tk}|{mk}|{a}/{m}"
        W = a * TR + m * MO
        eq, turn = simulate(W, close, starts, cfg.backtest, cfg.execution, cfg.breaker)
        s = summarize(eq, turn, st, PERIODS)
        rows.append(
            {
                "variant": name,
                **{f"{p}_{c}": s.loc[p, c] for p in PERIODS for c in ("score", "median_return")},
            }
        )
        daily[name] = continuous_daily_returns(W, close, cfg)
    df = pd.DataFrame(rows).sort_values("2024-25_score", ascending=False)
    print(df.round(4).to_string(index=False))
    res = pbo(pd.DataFrame(daily), n_partitions=16)
    print(
        f"\nPBO (CSCV, 16 partitions, {res['n_splits']} splits): {res['pbo']:.1%}  ({time.time() - t0:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
