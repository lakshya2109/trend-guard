"""Combine the candidates that pass the paired test in both periods, and score with the same test."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.harness import Lab  # noqa: E402
from research.jump_filter import apply as jump_apply  # noqa: E402
from research.paired import run  # noqa: E402

lab = Lab()
z = np.load(ROOT / "data/extra/variants.npz")
lp = np.log(lab.close.ffill(limit=3)).to_numpy()
r1 = np.vstack([np.zeros((1, lp.shape[1])), np.nan_to_num(np.diff(lp, axis=0))])
inv = z["momentum inverse-vol^0.5"]
books = {
    "jump +10%/24h": z["jump filter +10% _ 24h"],
    "inv-vol^0.5": inv,
    "inv-vol^0.5 + jump +10%/24h": jump_apply(np.nan_to_num(inv), r1, np.log1p(0.10), 24, False),
    "inv-vol^0.5 + jump +15%/24h": jump_apply(np.nan_to_num(inv), r1, np.log1p(0.15), 24, False),
}
with pd.option_context(
    "display.width", 250, "display.max_columns", 20, "display.float_format", "{:+.3f}".format
):
    print(run(lab, books, z["base"]))
lab.report({"final (base)": z["base"], **books})
