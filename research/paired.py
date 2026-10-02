"""One test for every candidate change: paired stationary block bootstrap of the daily-return Sharpe.

    python research/paired.py

Each variant and the base are run continuously from 2024-01-01 through the contest simulator (same costs
and trade rule). The bootstrap resamples the SAME days for both (mean block 10 days, so autocorrelation
and volatility clusters are kept), giving the distribution of the Sharpe difference. A variant counts as
an improvement only if P(dSharpe > 0) is high AND it does not lose in either period (2024-25, 2026).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.features import build  # noqa: E402
from research.harness import Lab  # noqa: E402
from research.jump_filter import apply as jump_apply  # noqa: E402
from research.partial_rebal import smooth  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

BLOCK, REPS, SEED = 10, 2000, 11


def stationary_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (geometric block lengths, wrap-around)."""
    out = np.empty(n, dtype=np.int64)
    i = rng.integers(n)
    for k in range(n):
        out[k] = i
        i = rng.integers(n) if rng.random() < 1 / BLOCK else (i + 1) % n
    return out


def sharpe(r: np.ndarray) -> np.ndarray:
    return r.mean(axis=-1) / r.std(axis=-1) * np.sqrt(365)


def paired(rb: pd.Series, rv: pd.Series) -> dict:
    a, b = rb.to_numpy(), rv.to_numpy()
    rng = np.random.default_rng(SEED)
    idx = np.stack([stationary_idx(len(a), rng) for _ in range(REPS)])
    d = sharpe(b[idx]) - sharpe(a[idx])
    out = {"dSR": sharpe(b) - sharpe(a), "dSR_p05": np.quantile(d, 0.05), "P(dSR>0)": (d > 0).mean()}
    for name, m in (("2024-25", rb.index.year < 2026), ("2026", rb.index.year == 2026)):
        out[f"dSR {name}"] = sharpe(b[m]) - sharpe(a[m])
    out["d_ann_ret"] = (b.mean() - a.mean()) * 365
    out["corr"] = np.corrcoef(a, b)[0, 1]
    return out


def with_patch(lab: Lab, cfg, patches: dict[str, Callable]) -> np.ndarray:
    """book_components with some module functions swapped in memory (restored afterwards)."""
    saved = {k: getattr(B, k) for k in patches}
    try:
        for k, f in patches.items():
            setattr(B, k, f)
        return sum(B.book_components(lab.close, lab.qv, cfg, True).values())
    finally:
        for k, f in saved.items():
            setattr(B, k, f)


def variants(lab: Lab) -> dict[str, np.ndarray]:
    b = lab.cfg.book
    W0 = np.nan_to_num(lab.book)
    lp = np.log(lab.close.ffill(limit=3)).to_numpy()
    r1 = np.vstack([np.zeros((1, lp.shape[1])), np.nan_to_num(np.diff(lp, axis=0))])
    orig_mm, orig_ts, orig_tr, orig_mo = B.momentum_member, B.tstat_score, B.trend_sleeve, B.momentum_sleeve

    def _invvol(W, f, p, power=0.5):
        vol_d = np.nan_to_num(f.vol_h * np.sqrt(B.H), nan=0.05)
        out = np.zeros_like(W)
        for t in np.where(np.abs(W).sum(1) > 0)[0]:
            w = W[t]
            g = np.abs(w).sum()
            inv = np.where(w != 0, 1.0 / np.maximum(vol_d[t], 1e-4) ** power, 0.0)
            nw = np.zeros_like(w)
            for side in (1, -1):
                m = np.sign(w) == side
                if m.any():
                    nw[m] = side * np.abs(w[m]).sum() * inv[m] / inv[m].sum()
            s0 = B._cc_scale(w / g, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)
            s1 = B._cc_scale(nw / g, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)
            out[t] = nw * (s1 / s0 if s0 > 0 else 1.0)
        return out

    oi = build(lab.close, lab.qv)["f"]["oi_x_px_72"]
    rk = oi.rank(axis=1, pct=True)
    z = (
        ((rk - rk.mean(axis=1).to_numpy()[:, None]) / rk.std(axis=1).to_numpy()[:, None])
        .fillna(0.0)
        .to_numpy()
    )

    def mm_oi(f, p, lb, sk, a, *args):
        B.tstat_score = lambda ff, l2, s2=0: orig_ts(ff, l2, s2) + 0.25 * z
        try:
            return orig_mm(f, p, lb, sk, a, *args)
        finally:
            B.tstat_score = orig_ts

    hours = (0, 6, 12, 18)

    def tr_stagger(close, qv, p, *a, **k):
        return np.mean([orig_tr(close, qv, p.model_copy(update={"rebalance_hour_utc": h})) for h in hours], 0)

    def mo_stagger(close, qv, p, *a, **k):
        return np.mean(
            [orig_mo(close, qv, p.model_copy(update={"select_hour_utc": h}), *a, **k) for h in hours], 0
        )

    cov_off = b.model_copy(
        update={
            "trend": b.trend.model_copy(
                update={"cov_hours": 168, "cov_shrink": 0.0, "cov_ewma_halflife_h": 0.0}
            )
        }
    )
    ss5 = b.model_copy(update={"momentum": b.momentum.model_copy(update={"short_share": 0.5})})
    return {
        "4x trend covariance (undo anant)": with_patch(lab, cov_off, {}),
        "jump filter +10% / 24h": jump_apply(W0, r1, np.log1p(0.10), 24, False),
        "jump filter +15% / 24h": jump_apply(W0, r1, np.log1p(0.15), 24, False),
        "jump filter +15% / 72h": jump_apply(W0, r1, np.log1p(0.15), 72, False),
        "momentum inverse-vol^0.5": with_patch(
            lab,
            b,
            {"momentum_member": lambda f, p, lb, sk, a, *x: _invvol(orig_mm(f, p, lb, sk, a, *x), f, p)},
        ),
        "momentum short share 0.5": with_patch(lab, ss5, {}),
        "momentum OI tilt 0.25": with_patch(lab, b, {"momentum_member": mm_oi}),
        "stagger both sleeves 00/06/12/18": with_patch(
            lab, b, {"trend_sleeve": tr_stagger, "momentum_sleeve": mo_stagger}
        ),
        "partial rebalance 0.5": smooth(W0, 0.5, True),
    }


def run(lab: Lab, books: dict[str, np.ndarray], base: np.ndarray) -> pd.DataFrame:
    rb = lab.daily(base)
    rows = {}
    for name, W in books.items():
        rows[name] = paired(rb, lab.daily(W))
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    lab = Lab()
    V = variants(lab)
    np.savez_compressed(
        ROOT / "data/extra/variants.npz", base=lab.book, **{k.replace("/", "_"): v for k, v in V.items()}
    )
    df = run(lab, V, lab.book)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(
            f"base = trend-guard-final (Sharpe {sharpe(lab.daily(lab.book).to_numpy()):.3f}); "
            f"{REPS} paired stationary-bootstrap reps, mean block {BLOCK} days"
        )
        print(df)
