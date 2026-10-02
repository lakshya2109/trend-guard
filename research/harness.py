"""Research harness: score any hourly target-weight matrix the way the contest does, plus one continuous run.

    from research.harness import Lab
    lab = Lab()                      # loads the panel and the live book once (cached to data/extra/)
    lab.report({"book": lab.book, "new": W})

Every number uses backtest/contest.py's simulator (fills at the hourly close, fee + slippage per side,
the live trade rule), so research results are comparable with docs/STRATEGY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backtest.contest import (  # noqa: E402
    _simulate,
    competition_metrics,
    non_overlapping,
    simulate,
    window_starts,
)
from backtest.run_backtest import load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import book_components  # noqa: E402

CACHE = ROOT / "data" / "extra" / "book_components.npz"


class Lab:
    def __init__(self, cost: float | None = None) -> None:
        self.cfg = load_strategy_config(ROOT / "config/strategy.yaml")
        sym = load_symbols(ROOT / "config/symbols.yaml")
        self.close, self.qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
        self.idx, self.cols = self.close.index, list(self.close.columns)
        self.px = self.close.ffill(limit=3).to_numpy(dtype=np.float64)
        bt = self.cfg.backtest
        self.cost = cost if cost is not None else bt.fee + bt.slippage
        self.starts = window_starts(self.idx, bt)
        self.comp = self._components()
        self.book = sum(self.comp.values())

    def _components(self) -> dict[str, np.ndarray]:
        key = f"{len(self.idx)}_{self.idx[-1].value}_{self.cfg.book.risk_multiplier}"
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=True)
            if str(z["key"]) == key:
                return {k: z[k] for k in z.files if k != "key"}
        comp = book_components(self.close, self.qv, self.cfg.book, self.cfg.book.allow_short)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, key=key, **comp)
        return comp

    # ------------------------------------------------------------------ simulation
    def continuous(self, W: np.ndarray, start: str = "2024-01-01 12:00") -> pd.Series:
        """One run from `start` to the end of the data; returns the hourly equity curve."""
        s0 = self.idx.get_loc(pd.Timestamp(start, tz="UTC"))
        H = len(self.idx) - 1 - s0
        ex = self.cfg.execution
        act = np.isin(self.idx.hour, ex.decision_hours_utc)
        eq, _, _ = _simulate(
            np.ascontiguousarray(np.nan_to_num(W), dtype=np.float64),
            self.px,
            act,
            np.array([s0], dtype=np.int64),
            H,
            self.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            1.0,
            9.0,
            1.0,
            9.0,  # breaker off: measure the raw stream
        )
        return pd.Series(eq[0], index=self.idx[s0 : s0 + H + 1])

    def windows(self, W: np.ndarray) -> pd.DataFrame:
        bt = self.cfg.backtest.model_copy(update={"fee": self.cost, "slippage": 0.0})
        eq, turnover, traded = simulate(
            np.nan_to_num(W),
            self.close,
            self.starts,
            bt,
            self.cfg.execution,
            self.cfg.breaker,
            with_trades=True,
        )
        m = competition_metrics(eq, traded, self.idx[self.starts])
        m["turnover"] = turnover
        return m

    # ------------------------------------------------------------------ reporting
    def stats(self, W: np.ndarray, name: str) -> dict:
        eq = self.continuous(W)
        d = eq.iloc[::24].pct_change().dropna()
        out: dict = {"name": name}
        for y in ("2024", "2025", "2026"):
            r = d[d.index.year == int(y)]
            out[f"SR{y[2:]}"] = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
        out["SRall"] = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0.0
        out["ann_ret"] = (1 + d.mean()) ** 365 - 1
        out["maxDD"] = float((1 - eq / eq.cummax()).max())
        gross = np.abs(np.nan_to_num(W)).sum(axis=1)
        out["gross"] = float(gross[self.idx >= pd.Timestamp("2024-01-01", tz="UTC")].mean())
        m = self.windows(W)
        sel = m.iloc[non_overlapping(m.index, 14)]
        out["fn_med"] = sel["return"].median()
        out["fn_mean"] = sel["return"].mean()
        out["P<-5%"] = (sel["return"] < -0.05).mean()
        out["P>5%"] = (sel["return"] > 0.05).mean()
        out["fn_sharpe"] = m["sharpe"].median()  # all overlapping windows: less noisy
        out["win_pos"] = (m["return"] > 0).mean()
        out["turn/fn"] = m["turnover"].median()
        return out

    def report(self, books: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = [self.stats(W, k) for k, W in books.items()]
        df = pd.DataFrame(rows).set_index("name")
        with pd.option_context(
            "display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format
        ):
            print(df)
        return df

    def daily(self, W: np.ndarray) -> pd.Series:
        return self.continuous(W).iloc[::24].pct_change().dropna()
