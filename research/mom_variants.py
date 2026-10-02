"""Momentum sleeve weighting variants (prototype of book.momentum_member with a weighting switch)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

orig = B.momentum_member


def member(f, p, lookbacks, skip, allow_short, weighting="equal", power=1.0):
    W = orig(f, p, lookbacks, skip, allow_short)
    if weighting == "equal":
        return W
    vol_d = np.nan_to_num(f.vol_h * np.sqrt(B.H), nan=0.05)
    out = np.zeros_like(W)
    for t in range(W.shape[0]):
        w = W[t]
        if not w.any():
            continue
        g = np.abs(w).sum()
        inv = np.where(w != 0, 1.0 / np.maximum(vol_d[t], 1e-4) ** power, 0.0)
        nw = np.zeros_like(w)
        for side in (1, -1):
            m = np.sign(w) == side
            if m.any():
                nw[m] = side * np.abs(w[m]).sum() * inv[m] / inv[m].sum()  # same side budget, inv-vol split
        # keep the sleeve on its vol target: rescale by the constant-correlation model
        s0 = B._cc_scale(w / g, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)
        s1 = B._cc_scale(nw / g, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)
        out[t] = nw * (s1 / s0 if s0 > 0 else 1.0)
    return out


if __name__ == "__main__":
    lab = Lab()
    cfg = lab.cfg.book
    rest = lab.book - lab.comp["v2_momentum"]
    scale = lab.comp["v2_momentum"]
    books = {"book": lab.book}
    base_m = B.momentum_sleeve(lab.close, lab.qv, cfg.momentum, True)
    for name, kw in {
        "invvol": {"weighting": "inv"},
        "invvol^0.5": {"weighting": "inv", "power": 0.5},
    }.items():
        B.momentum_member = lambda f, p, lb, sk, a, kw=kw: member(f, p, lb, sk, a, **kw)
        m = B.momentum_sleeve(lab.close, lab.qv, cfg.momentum, True)
        B.momentum_member = orig
        base = base_m
        # same capital share and risk dial as the live book (gross cap 1 applied afterwards)
        k = np.divide(
            np.abs(scale).sum(1), np.abs(base).sum(1), out=np.zeros(len(base)), where=np.abs(base).sum(1) > 0
        )[:, None]
        Wb = rest + m * k
        g = np.abs(Wb).sum(1, keepdims=True)
        books[f"book {name}"] = Wb / np.maximum(g, 1.0)
        books[f"v2_mom {name}"] = m
    books["v2_mom equal"] = base_m
    lab.report(books)
