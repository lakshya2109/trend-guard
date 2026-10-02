"""Event study: after a 1-hour jump above `thr`, the coin's return vs the market over the next h hours.

Entry at the close of the jump hour +1h (the bot acts 3 minutes after a bar closes, and a fill takes
time), so the reaction inside the next hour is not counted. Per year, with t-stats over events.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.features import build  # noqa: E402
from research.harness import Lab  # noqa: E402

lab = Lab()
d = build(lab.close, lab.qv, liq_min=2e6, min_age=336)
ok = d["elig"].to_numpy()
lp = np.log(lab.close.ffill(limit=3)).to_numpy()
r1 = np.vstack([np.full((1, lp.shape[1]), np.nan), np.diff(lp, axis=0)])
mkt = np.nanmean(np.where(ok, r1, np.nan), axis=1)
cm = np.nancumsum(np.nan_to_num(mkt))
idx = lab.idx
start = idx.searchsorted(pd.Timestamp("2023-09-01", tz="UTC"))
for thr in (0.07, 0.10, 0.15):
    for direction in (1, -1):
        rows = []
        T, N = lp.shape
        last = np.full(N, -10_000)
        for t in range(start, T - 80):
            for i in np.where(ok[t] & (direction * r1[t] > np.log1p(thr)))[0]:
                if t - last[i] < 72:  # one event per coin per 3 days
                    continue
                last[i] = t
                e = t + 1  # enter one bar later
                out = {"t": idx[t], "coin": lab.cols[i]}
                for h in (24, 72):
                    if e + h < T and np.isfinite(lp[e + h, i]) and np.isfinite(lp[e, i]):
                        out[f"ex{h}"] = (lp[e + h, i] - lp[e, i]) - (cm[e + h] - cm[e])
                        out[f"abs{h}"] = lp[e + h, i] - lp[e, i]
                rows.append(out)
        df = pd.DataFrame(rows).dropna()
        g = df.groupby(df.t.dt.year)
        tab = g[["ex24", "ex72", "abs72"]].mean() * 100
        tab["n"] = g.size()
        tab["t72"] = g["ex72"].mean() / g["ex72"].std() * np.sqrt(g.size())
        tab.loc["all"] = list(df[["ex24", "ex72", "abs72"]].mean() * 100) + [
            len(df),
            df.ex72.mean() / df.ex72.std() * np.sqrt(len(df)),
        ]
        tab["win%"] = list(g["ex72"].apply(lambda s: (s < 0).mean() * 100)) + [(df.ex72 < 0).mean() * 100]
        print(
            f"--- 1h move {'above +' if direction > 0 else 'below -'}{thr:.0%}: "
            "next-day / 3-day return vs market (%)"
        )
        with pd.option_context("display.float_format", "{:+.2f}".format):
            print(tab)
