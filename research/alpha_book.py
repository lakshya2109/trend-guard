"""Turn a (T, N) prediction panel into a market-neutral sleeve of hourly target weights.

At each decision hour: score = cross-sectional z of the (optionally smoothed) prediction;
weight ~ score / vol, optionally only the top/bottom k; dollar-neutral, optionally beta-neutral; scaled
to a daily volatility target with the trailing sample covariance and capped at gross_cap. Held until the
next decision.
"""

from __future__ import annotations

import numpy as np


def alpha_sleeve(
    pred: np.ndarray,
    ret_h: np.ndarray,
    vol_d: np.ndarray,
    beta: np.ndarray,
    hours: np.ndarray,
    decide_hours: tuple[int, ...] = (0,),
    smooth: int = 1,
    k: int | None = None,
    target_vol: float = 0.01,
    gross_cap: float = 1.0,
    beta_neutral: bool = True,
    long_only: bool = False,
    cov_hours: int = 168,
    deadband: float = 0.0,
) -> np.ndarray:
    T, N = pred.shape
    W = np.zeros((T, N))
    cur = np.zeros(N)
    hist: list[np.ndarray] = []
    for t in range(T):
        if hours[t] in decide_hours and np.isfinite(pred[t]).sum() >= 10:
            p = pred[t]
            ok = np.isfinite(p) & np.isfinite(vol_d[t]) & (vol_d[t] > 0)
            z = np.where(ok, p, np.nan)
            z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-12)
            hist.append(np.nan_to_num(z))
            hist = hist[-smooth:]
            s = np.mean(hist, axis=0)
            s[~ok] = 0.0
            if k:
                order = np.argsort(s)
                keep = np.zeros(N, dtype=bool)
                keep[order[-k:]] = True
                if not long_only:
                    keep[order[:k]] = True
                s = np.where(keep, s, 0.0)
            if long_only:
                s = np.clip(s, 0, None)
            w = np.where(ok, s / np.where(ok, vol_d[t], 1.0), 0.0)
            if not long_only:
                for _ in range(3):  # project out the dollar and beta exposure (only on held names)
                    held = w != 0
                    if not held.any():
                        break
                    w[held] -= w[held].mean()
                    if beta_neutral:
                        b = np.nan_to_num(beta[t], nan=1.0)
                        # beta-neutral by scaling the two sides: long beta exposure == short beta exposure
                        lb = (w[w > 0] * b[w > 0]).sum()
                        sb = (-w[w < 0] * b[w < 0]).sum()
                        if lb > 0 and sb > 0:
                            w[w > 0] *= np.sqrt(sb / lb)
                            w[w < 0] *= np.sqrt(lb / sb)
            held = w != 0
            if held.any():
                R = np.nan_to_num(ret_h[max(0, t - cov_hours + 1) : t + 1][:, held])
                C = np.atleast_2d(np.cov(R, rowvar=False)) * 24
                vol = float(np.sqrt(max(w[held] @ C @ w[held], 1e-12)))
                w *= min(target_vol / vol, gross_cap / np.abs(w).sum())
            if deadband > 0:  # keep the old weight where the change is small (turnover control)
                small = np.abs(w - cur) < deadband * np.maximum(np.abs(cur), 1e-9)
                w = np.where(small & (cur != 0) & (np.sign(w) == np.sign(cur)), cur, w)
            cur = w
        W[t] = cur
    return W
