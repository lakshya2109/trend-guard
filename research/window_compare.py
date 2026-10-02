"""Window-by-window comparison of two books on the competition metrics (all start dates, all 14 offsets)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.benchmark import rank_vs_field  # noqa: E402
from backtest.contest import non_overlapping  # noqa: E402
from research.harness import Lab  # noqa: E402

lab = Lab()
base = np.load(ROOT / "data/extra/variants.npz")["base"]  # trend-guard-final
ma, mb = lab.windows(base), lab.windows(lab.book)  # base vs this branch's live book
for m in (ma, mb):
    m["rank"] = [
        rank_vs_field(s, so, c, a)
        for s, so, c, a in zip(m.sharpe, m.sortino, m.calmar_total, m.active_days, strict=True)
    ]
cols = ["return", "sharpe", "sortino", "calmar_total", "max_drawdown"]
print("all", len(ma), "overlapping windows: median base / v2, share of windows where v2 is better")
for c in cols:
    better = (mb[c] < ma[c]) if c == "max_drawdown" else (mb[c] > ma[c])
    print(f"  {c:13s} {ma[c].median():+.4f} / {mb[c].median():+.4f}   v2 better in {better.mean():.0%}")
print(f"  P(rank 1 vs last edition) {np.mean(ma['rank'] == 1):.0%} / {np.mean(mb['rank'] == 1):.0%}")
rows = []
for off in range(14):
    sel = non_overlapping(ma.index, 14, off)
    rows.append(
        {
            c: mb[c].iloc[sel].median() - ma[c].iloc[sel].median()
            for c in ("return", "sharpe", "sortino", "calmar_total")
        }
    )
d = pd.DataFrame(rows)
print(
    "\nnon-overlapping medians, v2 minus base, for each of the 14 start offsets (offset 0 = official report):"
)
with pd.option_context("display.float_format", "{:+.4f}".format):
    print(d)
    print("offsets where v2 is higher:", (d > 0).sum().to_dict())
