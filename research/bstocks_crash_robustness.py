"""Robustness of the crash-only COIN/MSTR short (research/bstocks_crash_cushion.py): every combination
of hedge size (25/50/75% of net long crypto) and trigger (BTC 72h return below -5/-6/-8/-10%), per
period, against round 8's adoption rule, plus crash vs normal windows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import bstocks_crash_cushion as cx  # noqa: E402
import stocks_sleeve_experiment as sx  # noqa: E402
from backtest.contest import simulate  # noqa: E402

idx, close = cx.idx, cx.close
decide = pd.Series(np.isin(idx.hour, [0, 12]), index=idx)
starts = sx.starts
btc_dd = np.array([(cx.btc.iloc[s:s + 337] / cx.btc.iloc[s:s + 337].cummax() - 1).min() for s in starts])
crash = btc_dd <= -0.15


def hedge(thr, share):
    trig = pd.Series(np.where(decide, (cx.r72 < thr).reindex(idx).fillna(False).astype(float), np.nan), index=idx).ffill().fillna(0).to_numpy()
    W = np.zeros(close.shape); W[:, :cx.ncr] = cx.book
    for p in cx.BETA:
        W[:, close.columns.get_loc(p)] -= trig * cx.net_long * share / len(cx.BETA) * close[p].notna().to_numpy()
    g = np.abs(W).sum(1, keepdims=True)
    return np.nan_to_num(W * np.where(g > 1.0, 1.0 / np.maximum(g, 1e-12), 1.0)), trig.mean()


def crash_normal(W):
    eq, _ = simulate(W, close, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
    r = eq[:, -1] / eq[:, 0] - 1
    return np.median(r[crash]), r[crash].min(), np.median(r[~crash])


base = sx.evaluate(cx.variants["crypto only (live book)"]); bc = crash_normal(cx.variants["crypto only (live book)"])
fmt = lambda r, p: f"{r[p][0]:.3f} {r[p][2]:.0%} {r[p][3]:+.1%}"
print(f"{'':30s}{'on':>6s}{'2024-25 score >-1% worst':>26s}{'2026':>24s}{'all':>24s}{'crash med/worst':>18s}{'normal med':>11s}  rule")
print(f"{'crypto only':30s}{'':6s}{fmt(base,'2024-25'):>26s}{fmt(base,'2026'):>24s}{fmt(base,'all'):>24s}{bc[0]:+9.2%}/{bc[1]:+.1%}{bc[2]:+11.2%}")
for thr in (-0.05, -0.06, -0.08, -0.10):
    for share in (0.25, 0.5, 0.75):
        W, on = hedge(thr, share); r = sx.evaluate(W); c = crash_normal(W)
        ok = (r["2024-25"][0] > base["2024-25"][0] and r["2026"][0] > base["2026"][0]
              and all(r[p][2] >= base[p][2] - 0.02 for p in ("2024-25", "2026")) and r["all"][3] >= base["all"][3] - 1e-9)
        print(f"{f'BTC 3d < {thr:.0%}, hedge {share:.0%}':30s}{on:6.1%}{fmt(r,'2024-25'):>26s}{fmt(r,'2026'):>24s}{fmt(r,'all'):>24s}"
              f"{c[0]:+9.2%}/{c[1]:+.1%}{c[2]:+11.2%}  {'PASS' if ok else 'fail'}", flush=True)
