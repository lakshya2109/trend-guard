"""Performance metrics: Sharpe, Sortino, Calmar under three sampling/annualisation definitions,
plus composite = 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar (BUILD_SPEC.md Section 11.1).

The organizers have not published their exact sampling/annualisation convention (BUILD_SPEC.md
Section 16 unknown #1), so we always report all three definitions rather than picking one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-8
HOURS_PER_YEAR = 24 * 365
DAYS_PER_YEAR = 365


@dataclass
class MetricResult:
    sharpe: float
    sortino: float
    calmar: float
    composite: float
    total_return: float
    max_drawdown: float
    red_share: float
    denom_hit_eps: bool  # flag windows where a ratio's denominator was clamped at EPS


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = 1 - equity / peak
    return float(dd.max()) if len(dd) else 0.0


def _drawdown_of_returns(returns: pd.Series) -> float:
    """Max drawdown of the cumulative-return curve implied by a return series."""
    equity = (1 + returns).cumprod()
    return _max_drawdown(equity)


def _sortino(returns: pd.Series) -> tuple[float, bool]:
    downside = returns.clip(upper=0)
    denom = float(np.sqrt((downside**2).mean()))
    hit = denom < EPS
    return float(returns.mean() / max(denom, EPS)), hit


def _sharpe(returns: pd.Series) -> tuple[float, bool]:
    denom = float(returns.std(ddof=0))
    hit = denom < EPS
    return float(returns.mean() / max(denom, EPS)), hit


def _calmar(returns: pd.Series, mdd: float) -> tuple[float, bool]:
    hit = mdd < EPS
    return float(returns.mean() / max(mdd, EPS)), hit


def _red_share(returns: pd.Series) -> float:
    return float((returns < 0).mean()) if len(returns) else 0.0


def daily_raw(daily_returns: pd.Series) -> MetricResult:
    """Sharpe/Sortino: mean/std over raw daily returns. Calmar: mean daily return / max_dd."""
    sharpe, hit1 = _sharpe(daily_returns)
    sortino, hit2 = _sortino(daily_returns)
    mdd = _drawdown_of_returns(daily_returns)
    calmar, hit3 = _calmar(daily_returns, mdd)
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    total_return = float((1 + daily_returns).prod() - 1)
    return MetricResult(
        sharpe, sortino, calmar, composite, total_return, mdd, _red_share(daily_returns), hit1 or hit2 or hit3
    )


def daily_ann(daily_returns: pd.Series) -> MetricResult:
    """Sharpe/Sortino annualised by sqrt(365). Calmar: CAGR / max_dd."""
    sharpe, hit1 = _sharpe(daily_returns)
    sharpe *= np.sqrt(DAYS_PER_YEAR)
    sortino, hit2 = _sortino(daily_returns)
    sortino *= np.sqrt(DAYS_PER_YEAR)
    mdd = _drawdown_of_returns(daily_returns)
    n = len(daily_returns)
    total_return = float((1 + daily_returns).prod() - 1)
    cagr = float((1 + total_return) ** (DAYS_PER_YEAR / n) - 1) if n > 0 else 0.0
    hit3 = mdd < EPS
    calmar = cagr / max(mdd, EPS)
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    hit = hit1 or hit2 or hit3
    return MetricResult(sharpe, sortino, calmar, composite, total_return, mdd, _red_share(daily_returns), hit)


def hourly_raw(hourly_returns: pd.Series) -> MetricResult:
    """Same shape as daily_raw but over hourly returns (red_share is red-hour share here)."""
    sharpe, hit1 = _sharpe(hourly_returns)
    sortino, hit2 = _sortino(hourly_returns)
    mdd = _drawdown_of_returns(hourly_returns)
    calmar, hit3 = _calmar(hourly_returns, mdd)
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
    total_return = float((1 + hourly_returns).prod() - 1)
    hit = hit1 or hit2 or hit3
    red = _red_share(hourly_returns)
    return MetricResult(sharpe, sortino, calmar, composite, total_return, mdd, red, hit)


def turnover(notional_traded: pd.Series, equity: pd.Series) -> float:
    """Sum of traded notional divided by mean equity over the period -- a simple turnover proxy."""
    mean_eq = float(equity.mean()) if len(equity) else 0.0
    return float(notional_traded.sum() / mean_eq) if mean_eq > EPS else 0.0


def all_definitions(daily_returns: pd.Series, hourly_returns: pd.Series) -> dict[str, MetricResult]:
    """Convenience wrapper returning all three definitions keyed by name, for reports/dashboard."""
    return {
        "daily_raw": daily_raw(daily_returns),
        "daily_ann": daily_ann(daily_returns),
        "hourly_raw": hourly_raw(hourly_returns),
    }
