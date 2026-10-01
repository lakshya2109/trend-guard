"""The previous edition's final leaderboard (Composite Scores.pdf, 52 teams, $50k each) as a yardstick.

Their score is  active_days x (0.4 Sortino_n + 0.3 Sharpe_n + 0.3 Calmar_n),  where each ratio is
normalised across the field (best team = 1). It reproduces the published scores of the top rows to
rounding (e.g. 11 x (0.4 x 0.97 + 0.3 x 0.95 + 0.3 x 0.93) = 10.47 vs 10.50). Their Calmar matches
total return / max drawdown (a team that ends at its low scores -1). The exact normalisation is not
published; a percentile rank orders the field like the published scores (Spearman 0.97), so ranks
here are estimates. The field traded one past market: a rank says how a fortnight's ratios compare
with last year's teams, not where the bot would finish against this year's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sharpe, Sortino, Calmar, active days, published score (rows in published order)
FIELD = np.array(
    [
        [0.19, 0.25, 0.95, 11, 11.00],
        [0.13, 0.21, 0.75, 11, 10.50],
        [-0.02, -0.02, -0.09, 15, 10.43],
        [-0.11, -0.15, -0.57, 15, 8.16],
        [-0.15, -0.19, -0.49, 15, 8.04],
        [-0.23, -0.18, -0.83, 15, 7.09],
        [-0.19, -0.15, -0.77, 14, 6.96],
        [-0.15, -0.19, -0.86, 14, 6.81],
        [-0.22, -0.15, -0.81, 14, 6.81],
        [-0.28, -0.23, -0.99, 15, 6.40],
        [-0.37, -0.33, -0.87, 15, 5.86],
        [-0.31, -0.29, -0.86, 14, 5.82],
        [-0.17, -0.13, -0.65, 11, 5.77],
        [-0.37, -0.33, -0.95, 15, 5.69],
        [-0.30, -0.44, -0.83, 15, 5.67],
        [-0.39, -0.38, -0.83, 15, 5.62],
        [-0.39, -0.38, -0.83, 15, 5.61],
        [-0.39, -0.38, -0.83, 15, 5.58],
        [-0.21, -0.16, -0.70, 11, 5.47],
        [-0.41, -0.37, -0.98, 15, 5.38],
        [-0.40, -0.41, -0.97, 15, 5.24],
        [-0.27, -0.27, -0.88, 12, 5.15],
        [-0.45, -0.42, -0.93, 15, 5.09],
        [-0.45, -0.42, -0.92, 15, 5.09],
        [-0.42, -0.36, -0.92, 14, 5.08],
        [-0.40, -0.42, -0.88, 14, 4.98],
        [-0.44, -0.39, -0.89, 14, 4.94],
        [-0.46, -0.47, -0.95, 15, 4.81],
        [-0.49, -0.46, -0.96, 15, 4.77],
        [-0.30, -0.22, -1.00, 11, 4.67],
        [-0.50, -0.52, -0.88, 15, 4.60],
        [-0.41, -0.59, -0.96, 15, 4.52],
        [-0.50, -0.46, -0.93, 14, 4.44],
        [-0.48, -0.50, -0.94, 14, 4.35],
        [-0.31, -0.25, -0.80, 10, 4.32],
        [-0.61, -0.58, -0.97, 15, 3.92],
        [-0.59, -0.62, -1.00, 15, 3.80],
        [-0.64, -0.69, -0.97, 15, 3.47],
        [-0.67, -0.75, -0.96, 15, 3.25],
        [-0.49, -0.64, -1.00, 12, 3.25],
        [-0.36, -0.28, -1.00, 8, 3.15],
        [-0.83, -0.82, -1.00, 15, 2.62],
        [-0.65, -0.66, -1.00, 11, 2.59],
        [-0.90, -1.02, -1.00, 15, 2.03],
        [-0.84, -1.12, -0.98, 15, 1.98],
        [-1.01, -1.01, -1.00, 15, 1.86],
        [-0.44, -0.33, -1.00, 5, 1.81],
        [-0.99, -1.11, -0.98, 14, 1.60],
        [-1.11, -1.23, -0.99, 15, 1.33],
        [-0.25, -5.50, -1.00, 2, 0.37],
        [-0.26, -9.82, -1.00, 2, 0.37],
        [-1.80, -1.98, -10.0, 15, 0.18],
    ]
)
WINNER = {"sharpe": 0.19, "sortino": 0.25, "calmar_total": 0.95, "active_days": 11}  # Team70, $52,506


def _pct_rank(x: np.ndarray) -> np.ndarray:
    """0 = worst, 1 = best (ties share their average rank)."""
    r = pd.Series(x).rank(method="average").to_numpy()
    return (r - 1) / max(len(x) - 1, 1)


def field_scores(
    sharpe: np.ndarray, sortino: np.ndarray, calmar: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """active_days x (0.4 Sortino + 0.3 Sharpe + 0.3 Calmar), each ratio percentile-normalised."""
    return active * (0.4 * _pct_rank(sortino) + 0.3 * _pct_rank(sharpe) + 0.3 * _pct_rank(calmar))


def rank_vs_field(sharpe: float, sortino: float, calmar: float, active: float) -> int:
    """Where one fortnight would have finished among the previous edition's 52 teams (1 = first)."""
    sc = field_scores(
        np.r_[FIELD[:, 0], sharpe],
        np.r_[FIELD[:, 1], sortino],
        np.r_[FIELD[:, 2], calmar],
        np.r_[FIELD[:, 3], active],
    )
    return int(1 + (sc[:-1] > sc[-1]).sum())
