"""Quintile sorts: mean forward return by quintile of a feature (equal weight), per year."""

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
px = np.exp(lp)
idx = lab.idx
h = int(sys.argv[2]) if len(sys.argv) > 2 else 168
rows = np.where((idx >= pd.Timestamp("2024-01-01", tz="UTC")) & (idx.hour == 0))[0][:: max(1, h // 24)]
fwd = np.full_like(px, np.nan)
fwd[:-h] = px[h:] / px[:-h] - 1  # simple return (means, not logs)
for feat in sys.argv[1].split(","):
    X = d["f"][feat].to_numpy()
    res = []
    for t in rows:
        m = ok[t] & np.isfinite(X[t]) & np.isfinite(fwd[t])
        if m.sum() < 15:
            continue
        q = pd.qcut(pd.Series(X[t, m]).rank(method="first"), 5, labels=False).to_numpy()
        f = fwd[t, m]
        res.append([idx[t].year] + [f[q == i].mean() for i in range(5)] + [f.mean()])
    df = pd.DataFrame(res, columns=["y", "Q1", "Q2", "Q3", "Q4", "Q5", "mkt"])
    df["Q1-Q5"] = df.Q1 - df.Q5
    g = df.groupby("y").mean()
    g.loc["all"] = df.mean().drop("y")
    g["t(Q1-Q5)"] = list(df.groupby("y")["Q1-Q5"].apply(lambda s: s.mean() / s.std() * np.sqrt(len(s)))) + [
        df["Q1-Q5"].mean() / df["Q1-Q5"].std() * np.sqrt(len(df))
    ]
    print(f"--- {feat}, forward {h}h, non-overlapping, mean simple returns (Q1 = lowest feature)")
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(g)
