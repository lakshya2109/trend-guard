"""Robustness of the crowded-long filter (drop momentum longs while 3-day funding > +thr)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import funding_experiment as fx  # noqa: E402  (reuses its panel, funding feature and components)
from backtest.contest import contest_score, simulate  # noqa: E402

def per_window(W):
    eq, _ = fx.simulate(W, fx.close, fx.starts, fx.cfg.backtest, fx.cfg.execution, fx.cfg.breaker)
    return pd.DataFrame({"ret": eq[:, -1] / eq[:, 0] - 1, "score": contest_score(eq)}, index=fx.st)

base = per_window(fx.variant(1.0))
F, comp = fx.F, fx.comp
mom = comp["v2_momentum"] + comp["team_momentum"]
yr = base.index.year
print("how often the filter binds (share of hours with a momentum long in a coin whose 3d funding > thr):")
for thr in (0.0002, 0.0003, 0.0004, 0.0005):
    hot = (mom > 0) & (F > thr)
    by = pd.Series(hot.any(axis=1), index=fx.close.index).groupby(fx.close.index.year).mean()
    print(f"  {thr*1e4:.0f}bp: " + "  ".join(f"{y}: {v:.1%}" for y, v in by.items() if y >= 2024))
print()
print(f"{'variant':38s}{'2024':>8s}{'2025':>8s}{'2026':>8s}{'all':>8s}   windows better / worse   mean d-ret")
def show(name, W):
    r = per_window(W); med = lambda m: r.score[m].median() - base.score[m].median()
    d = r.ret - base.ret
    print(f"{name:38s}{med(yr == 2024):+8.3f}{med(yr == 2025):+8.3f}{med(yr == 2026):+8.3f}{med(yr > 0):+8.3f}   {(d > 1e-6).mean():6.0%} / {(d < -1e-6).mean():4.0%}        {d.mean():+.3%}")
for thr in (0.0002, 0.0003, 0.0004, 0.0005):
    show(f"momentum longs, funding > {thr*1e4:.0f}bp", fx.variant(thr, longs_in=fx.MOM))
show("v2 momentum only, > 3bp", fx.variant(0.0003, longs_in=("v2_momentum",)))
show("team momentum only, > 3bp", fx.variant(0.0003, longs_in=("team_momentum",)))
show("momentum + trend longs, > 3bp", fx.variant(0.0003, longs_in=fx.MOM + fx.TREND))
print("\n(score columns: change in median window score vs the live book, by year of window start)")
