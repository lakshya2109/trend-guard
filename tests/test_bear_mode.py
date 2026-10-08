"""Bear-mode regime switch: hysteresis, causality, and that the switch only swaps whole books."""

import numpy as np
import pandas as pd

from trend_guard.config import BearModeConfig
from trend_guard.strategy.book import bear_regime


def market(path: list[float], n_coins: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=len(path), freq="1h", tz="UTC")
    close = pd.DataFrame({f"C{i}/USD": path for i in range(n_coins)}, index=idx)
    return close, pd.DataFrame(1e9, index=idx, columns=close.columns)  # all liquid


P = BearModeConfig(enabled=True, lookback_h=2, enter=-0.08, exit=-0.02, min_liquidity_usd=1.0)


def test_enters_below_enter_and_leaves_only_above_exit():
    path = [100, 100, 100] + [100] * 165 + [100, 95, 90, 89, 88, 88, 89, 87.5, 88, 92, 95]
    close, qv = market(path)
    on = bear_regime(close, qv, P)
    k = 168  # first hour with a full 7-day liquidity window
    # 2-hour returns from k: 0, -5%, -10% (enter), -6.3%, -2.2%, -1.1% (exit above -2%) ...
    assert not on[k + 1] and on[k + 2] and on[k + 3] and on[k + 4]  # hysteresis keeps it on at -2.2%
    assert not on[k + 5]  # -1.1% > -2%: back to normal


def test_regime_is_causal():
    rng = np.random.default_rng(1)
    path = list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400))))
    close, qv = market(path)
    full = bear_regime(close, qv, P)
    cut = bear_regime(close.iloc[:300], qv.iloc[:300], P)
    assert (full[:300] == cut).all()
