"""Walk-forward cross-sectional alpha model: LightGBM (and ridge) on the research feature panel.

    python research/alpha_model.py --label y_24 --tag base

Protocol (no look-ahead):
  - training rows are sampled every `step` hours on eligible coins; features are ranked across coins
    at each hour (market-state features are left raw);
  - the model is refit at the start of every month on all rows whose label window ended at least
    `embargo` hours before that month began (expanding window), then predicts every hour of the month;
  - first predicted month: 2024-01 (the first 7 months only train).
Writes data/extra/pred_<tag>.npz with the (T, N) prediction panel and prints yearly rank ICs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.features import build  # noqa: E402

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=400,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    num_threads=12,
)


def xs_rank(a: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Rank each row across eligible columns into [-0.5, 0.5]; NaN elsewhere."""
    x = np.where(ok, a, np.nan)
    r = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()
    n = np.sum(np.isfinite(x), axis=1, keepdims=True)
    return np.where(n > 2, r - 0.5 - 0.5 / np.maximum(n, 1), np.nan)


def panel_arrays(
    close: pd.DataFrame, qv: pd.DataFrame, drop: tuple[str, ...] = (), raw: bool = False
) -> dict:
    d = build(close, qv)
    ok = d["elig"].to_numpy()
    names, X = [], []
    for k, v in d["f"].items():
        if any(k.startswith(p) for p in drop):
            continue
        a = v.to_numpy(dtype=np.float64, copy=True)
        a[~np.isfinite(a)] = np.nan
        X.append(a if (k.startswith("m_") or raw) else xs_rank(a, ok))  # raw keeps time-series levels
        names.append(k)
    return {
        "X": np.stack(X, axis=2).astype(np.float32),  # (T, N, F)
        "names": names,
        "ok": ok,
        "y": {k: v.to_numpy() for k, v in d["y"].items()},
        "vol_d": d["vol_d"].to_numpy(),
        "beta": d["beta"].to_numpy(),
        "index": close.index,
    }


def walk_forward(
    P: dict, label: str, step: int, embargo: int, model: str, first: str = "2024-01-01"
) -> np.ndarray:
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    X, ok, idx = P["X"], P["ok"], P["index"]
    y = P["y"][label]
    T, N, F = X.shape
    h = int(label.split("_")[1])
    pred = np.full((T, N), np.nan)
    months = pd.date_range(pd.Timestamp(first, tz="UTC"), idx[-1], freq="MS")
    rows_t = np.arange(0, T, step)
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        t0, t1 = idx.searchsorted(m0), idx.searchsorted(m1)
        cut = t0 - h - embargo  # last row whose label is fully known before the month
        tr = rows_t[rows_t < cut]
        Xtr = X[tr].reshape(-1, F)
        ytr = y[tr].reshape(-1)
        keep = ok[tr].reshape(-1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[keep], np.clip(ytr[keep], -5, 5)  # winsorise fat-tailed labels
        Xte = X[t0:t1].reshape(-1, F)
        if model == "lgb":
            booster = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=300)
            p = booster.predict(Xte)
        else:
            Xf = np.nan_to_num(Xtr)
            p = Ridge(alpha=1000.0).fit(Xf, ytr).predict(np.nan_to_num(Xte))
        pred[t0:t1] = p.reshape(t1 - t0, N)
    pred[~ok] = np.nan
    return pred


def ic_table(pred: np.ndarray, P: dict, label: str, hour: int = 0) -> pd.DataFrame:
    idx = P["index"]
    y = P["y"][label]
    rows = []
    for t in np.where((idx.hour == hour) & np.isfinite(pred).any(axis=1))[0]:
        m = np.isfinite(pred[t]) & np.isfinite(y[t])
        if m.sum() < 10:
            continue
        a, b = pd.Series(pred[t, m]).rank(), pd.Series(y[t, m]).rank()
        rows.append((idx[t], a.corr(b), np.corrcoef(pred[t, m], np.clip(y[t, m], -5, 5))[0, 1]))
    df = pd.DataFrame(rows, columns=["t", "rank_ic", "ic"]).set_index("t")
    g = df.groupby(df.index.year)
    out = g.mean()
    out["ic_t"] = g["rank_ic"].mean() / g["rank_ic"].std() * np.sqrt(g.size())
    out.loc["all"] = [
        df.rank_ic.mean(),
        df.ic.mean(),
        df.rank_ic.mean() / df.rank_ic.std() * np.sqrt(len(df)),
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="y_24")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--embargo", type=int, default=24)
    ap.add_argument("--model", default="lgb")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--drop", default="")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    from research.harness import Lab

    lab = Lab()
    t = time.time()
    P = panel_arrays(lab.close, lab.qv, tuple(x for x in a.drop.split(",") if x), a.raw)
    print(f"features {P['X'].shape} in {time.time() - t:.0f}s: {P['names']}")
    t = time.time()
    pred = walk_forward(P, a.label, a.step, a.embargo, a.model)
    print(f"walk-forward {time.time() - t:.0f}s")
    print(ic_table(pred, P, a.label))
    np.savez_compressed(ROOT / "data" / "extra" / f"pred_{a.tag}.npz", pred=pred, names=np.array(P["names"]))


if __name__ == "__main__":
    main()
