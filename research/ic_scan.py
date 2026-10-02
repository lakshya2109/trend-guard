"""Univariate cross-sectional rank IC of every feature against forward residual returns."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.features import build  # noqa: E402
from research.harness import Lab  # noqa: E402

lab = Lab()
d = build(lab.close, lab.qv)
ok = d["elig"].to_numpy()
lp = np.log(lab.close.ffill(limit=3)).to_numpy()
idx = lab.idx
rows = np.where(idx >= pd.Timestamp("2024-01-01", tz="UTC"))[0][::6]  # every 6h
out = {}
for h in (1, 4, 12, 24, 72):
    fwd = np.full_like(lp, np.nan)
    fwd[:-h] = lp[h:] - lp[:-h]
    fwd = np.where(ok, fwd, np.nan)
    fwd = fwd - np.nanmean(fwd, axis=1, keepdims=True)
    Y = pd.DataFrame(fwd[rows]).rank(axis=1)
    for k, v in d["f"].items():
        if k.startswith("m_"):
            continue
        X = pd.DataFrame(np.where(ok, v.to_numpy(), np.nan)[rows]).rank(axis=1)
        ic = X.corrwith(Y, axis=1)
        yr = pd.Series(ic.to_numpy(), index=idx[rows]).groupby(idx[rows].year).mean()
        out.setdefault(k, {})[f"h{h}"] = ic.mean()
        if h == 24:
            for y, val in yr.items():
                out[k][f"h24_{y}"] = val
df = pd.DataFrame(out).T
with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 200):
    print(df.sort_values("h24"))
