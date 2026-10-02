"""Fama-MacBeth: daily cross-sectional regressions of forward returns on feature ranks."""

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
px = lab.close.ffill(limit=3).to_numpy()
idx = lab.idx
h = int(sys.argv[2]) if len(sys.argv) > 2 else 24
feats = sys.argv[1].split(",")
rows = np.where((idx >= pd.Timestamp("2024-01-01", tz="UTC")) & (idx.hour == 0))[0][:: max(1, h // 24)]
fwd = np.full_like(px, np.nan)
fwd[:-h] = px[h:] / px[:-h] - 1
X = {f: d["f"][f].to_numpy() for f in feats}
coefs = []
for t in rows:
    m = ok[t] & np.isfinite(fwd[t])
    for f in feats:
        m &= np.isfinite(X[f][t])
    if m.sum() < 15:
        continue
    A = np.column_stack(
        [np.ones(m.sum())] + [pd.Series(X[f][t, m]).rank(pct=True).to_numpy() - 0.5 for f in feats]
    )
    y = np.clip(fwd[t, m], -0.5, 0.5)  # simple returns, mild winsor only for data errors
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    coefs.append([idx[t].year] + list(b[1:]))
C = pd.DataFrame(coefs, columns=["y"] + feats)
out = {}
for y, g in list(C.groupby("y")) + [("all", C)]:
    out[y] = {
        f: f"{g[f].mean() * 1e4:+.1f}bp (t {g[f].mean() / g[f].std() * np.sqrt(len(g)):+.2f})" for f in feats
    }
print(f"forward {h}h: coefficient = return spread top-minus-bottom rank, per period")
print(pd.DataFrame(out).T.to_string())
