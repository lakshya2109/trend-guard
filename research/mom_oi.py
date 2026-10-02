"""Momentum sleeve ranked on momentum + beta x (cross-sectional z of OI-confirmed momentum, oi_x_px_72)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.features import build  # noqa: E402
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

lab = Lab()
oi = build(lab.close, lab.qv)["f"]["oi_x_px_72"]
r = oi.rank(axis=1, pct=True)
z = ((r - r.mean(axis=1).to_numpy()[:, None]) / r.std(axis=1).to_numpy()[:, None]).fillna(0.0).to_numpy()
orig = B.tstat_score
cfg = lab.cfg.book
base_m = B.momentum_sleeve(lab.close, lab.qv, cfg.momentum, True)
rest = lab.book - lab.comp["v2_momentum"]
books = {"book": lab.book}
for beta in (0.25, 0.5, 1.0):
    B.tstat_score = lambda f, lb, skip=0, beta=beta: orig(f, lb, skip) + beta * z  # momentum score + OI tilt
    m = B.momentum_sleeve(lab.close, lab.qv, cfg.momentum, True)
    B.tstat_score = orig
    k = np.divide(
        np.abs(lab.comp["v2_momentum"]).sum(1),
        np.abs(base_m).sum(1),
        out=np.zeros(len(m)),
        where=np.abs(base_m).sum(1) > 0,
    )[:, None]
    W = rest + m * k
    books[f"book +OI beta={beta}"] = W / np.maximum(np.abs(W).sum(1, keepdims=True), 1.0)
lab.report(books)
