"""Probability of Backtest Overfitting via combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017). BUILD_SPEC.md Section 11.3.

Given a T x N matrix of daily returns (T days, N configurations), split the rows into S equal
partitions. For every way of choosing S/2 partitions as in-sample (IS), the rest are out-of-sample
(OOS): pick the config with the best IS Sharpe and record its relative rank OOS. PBO is the share of
splits where the IS winner lands in the bottom half OOS (logit <= 0).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

EPS = 1e-12


def _sharpe(block: np.ndarray) -> np.ndarray:
    return block.mean(axis=0) / (block.std(axis=0) + EPS)


def pbo(returns: pd.DataFrame, n_partitions: int = 16) -> dict:
    """Returns {"pbo": float, "logits": np.ndarray, "n_splits": int}. Requires >= 2 configs."""
    m = returns.dropna().to_numpy(dtype=float)
    t, n = m.shape
    if n < 2 or t < n_partitions * 2:
        return {"pbo": float("nan"), "logits": np.array([]), "n_splits": 0}
    rows = t - (t % n_partitions)
    parts = np.array_split(m[:rows], n_partitions)
    logits = []
    for is_idx in combinations(range(n_partitions), n_partitions // 2):
        oos_idx = [i for i in range(n_partitions) if i not in is_idx]
        is_perf = _sharpe(np.vstack([parts[i] for i in is_idx]))
        oos_perf = _sharpe(np.vstack([parts[i] for i in oos_idx]))
        best = int(np.argmax(is_perf))
        # relative OOS rank of the IS winner in (0, 1): higher is better
        rank = (oos_perf < oos_perf[best]).sum() + 1
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1 - omega)))
    arr = np.array(logits)
    return {"pbo": float((arr <= 0).mean()), "logits": arr, "n_splits": len(arr)}
