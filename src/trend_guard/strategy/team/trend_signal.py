# Vendored UNMODIFIED from github.com/Gunek-945/roostoo-trend-book (bot/strategy.py at commit 93f91c4),
# written by teammate Gunek-945. Only this header was added; excluded from lint to stay his code.

"""Stateless signal layer: every value at hour t uses only bars that closed at or before t.

Signal = time-series momentum averaged over several horizons, each normalised by the
asset's own volatility (a t-statistic of the recent trend). Averaging horizons avoids
betting on one lookback that happened to fit the past.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalParams:
    lookbacks: tuple[int, ...] = (168, 336, 720)  # hours: 7, 14, 30 days
    vol_window: int = 168
    min_history: int = 336
    min_quote_volume_24h: float = 2_000_000.0
    score_cap: float = 2.0


@dataclass
class Features:
    score: pd.DataFrame       # trend strength in [-1, 1], NaN where not tradable
    vol_daily: pd.DataFrame   # daily volatility estimate from hourly returns
    returns: pd.DataFrame     # hourly log returns (for covariance)

    def row(self, t) -> dict:
        return {"score": self.score.loc[t], "vol_daily": self.vol_daily.loc[t]}


def compute_features(close: pd.DataFrame, quote_volume: pd.DataFrame, p: SignalParams) -> Features:
    close = close.ffill(limit=3)
    logp = np.log(close)
    ret = logp.diff()
    vol_h = ret.rolling(p.vol_window, min_periods=p.vol_window // 2).std()

    tstat = sum((logp - logp.shift(h)) / (vol_h * np.sqrt(h)) for h in p.lookbacks) / len(p.lookbacks)
    score = tstat.clip(-p.score_cap, p.score_cap) / p.score_cap

    liquid = quote_volume.rolling(24, min_periods=20).sum() >= p.min_quote_volume_24h
    seasoned = close.notna().rolling(p.min_history, min_periods=1).sum() >= p.min_history
    eligible = liquid & seasoned & vol_h.notna() & close.notna()

    return Features(score=score.where(eligible), vol_daily=vol_h * np.sqrt(24), returns=ret)
