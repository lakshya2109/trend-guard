"""Bear-mode exit: would staying in the rebound book longer have saved 12-13 Oct, and does it help in general?

Proposal (team, 14 Oct): bear mode left at 00:00 UTC on 12 Oct when the market's 5-day return crossed -2%, and
the normal book then lost on 13 Oct; stay in the rebound book until the market has really recovered. Tested
with the live book as now configured (net hedge on), bear mode on with exit at -2% (as it ran), 0%, +2%,
+4% or never (enter stays -8%), against bear mode off. Same simulator, costs and trade rule:
  1. contest replay from 4 Oct 12:00 and from the 12 Oct exit, with when each version left bear mode;
  2. continuous run from 2024, paired bootstrap of the daily Sharpe vs bear mode off (research/paired.py).

    python research/bear_exit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from research.paired import paired  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

lab = Lab()
b, bm = lab.cfg.book, lab.cfg.book.bear_mode
assert not bm.enabled, "the base is the deployed book (bear mode off)"
base = np.nan_to_num(lab.book)
EXITS = {"exit -2% (as it ran)": -0.02, "exit 0%": 0.0, "exit +2%": 0.02, "exit +4%": 0.04, "never exit": 9.0}


def with_exit(x: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": True, "exit": x})})
    W = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, cfg, True).values()))
    return W, B.bear_regime(lab.close, lab.qv, cfg.bear_mode)


V = {"bear mode off (deployed)": (base, np.zeros(len(lab.idx), bool))} | {
    k: with_exit(x) for k, x in EXITS.items()
}
c0 = lab.idx.get_loc(pd.Timestamp("2026-10-04 12:00", tz="UTC"))
d_base = lab.daily(base)
rows = []
for name, (W, on) in V.items():
    full = lab.continuous(W, "2026-10-04 12:00")
    tail = lab.continuous(W, "2026-10-12 00:00")
    left = lab.idx[c0:][np.diff(on[c0 - 1 :].astype(int)) == -1]
    row = {
        "book": name,
        "left bear mode": f"{left[0]:%m-%d %H:00}"
        if len(left)
        else ("-" if not on[c0:].any() else "still on"),
        "contest": f"{full.iloc[-1] / full.iloc[0] - 1:+.2%}",
        "12 Oct 00:00 -> now": f"{tail.iloc[-1] / tail.iloc[0] - 1:+.2%}",
    }
    if name != "bear mode off (deployed)":
        p = paired(d_base, lab.daily(W))
        row |= {
            "dSharpe vs off": f"{p['dSR']:+.2f}",
            "P(better)": f"{p['P(dSR>0)']:.0%}",
            "2024-25 / 2026": f"{p['dSR 2024-25']:+.2f} / {p['dSR 2026']:+.2f}",
            "d return/yr": f"{p['d_ann_ret']:+.1%}",
        }
    rows.append(row)
print(f"data to {lab.idx[-1]:%Y-%m-%d %H:00} UTC")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(pd.DataFrame(rows).set_index("book").fillna("").to_string())
