# Vendored UNMODIFIED from github.com/Gunek-945/roostoo-trend-book (bot/portfolio.py at commit 93f91c4),
# written by teammate Gunek-945. Only this header was added; excluded from lint to stay his code.

"""Stateful portfolio construction shared by the backtest and the live bot.

Once a day: weight every tradable asset by trend strength / volatility, then scale the
book to a target daily volatility using the recent covariance (crypto is highly
correlated, so this matters). Every hour: trailing stops per position and an equity
drawdown brake that de-risks before a loss can dominate the Calmar ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioParams:
    target_vol_daily: float = 0.004   # low on purpose: score ratios are scale-free, qualifying is not
    min_score: float = 0.25           # ignore trends weaker than this
    allow_shorts: bool = False        # tested: shorts lowered results in every variant
    max_weight: float = 0.5
    min_weight: float = 0.01
    gross_cap: float = 1.0
    cov_hours: int = 168
    rebalance_hour_utc: int = 12
    stop_vol_mult: float = 3.0        # trailing stop distance in daily-vol units (0 disables)
    dd_start: float = 0.05            # catastrophe brake only: tighter brakes sold dips and hurt
    dd_full: float = 0.10
    brake_step: float = 0.25          # brake moves in steps so small equity wiggles don't trade
    day_anchor_hour_utc: int = 0      # daily P&L lock resets at this hour
    day_take_profit: float = 0.0      # flat for the rest of the day once up this much (0 disables)
    day_stop: float = 0.0             # flat for the rest of the day once down this much (0 disables)


@dataclass
class PortfolioState:
    base: dict[str, float] = field(default_factory=dict)      # weights set at the last rebalance
    extreme: dict[str, float] = field(default_factory=dict)   # best price since entry
    stopped: list[str] = field(default_factory=list)          # flat until the next rebalance
    peak_equity: float = 0.0
    day_key: str = ""                                          # current P&L-lock day
    day_start_equity: float = 0.0
    day_locked: bool = False

    def to_dict(self) -> dict:
        return {"base": self.base, "extreme": self.extreme, "stopped": self.stopped,
                "peak_equity": self.peak_equity, "day_key": self.day_key,
                "day_start_equity": self.day_start_equity, "day_locked": self.day_locked}

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        return cls(base=dict(d.get("base", {})), extreme=dict(d.get("extreme", {})),
                   stopped=list(d.get("stopped", [])), peak_equity=float(d.get("peak_equity", 0.0)),
                   day_key=str(d.get("day_key", "")), day_start_equity=float(d.get("day_start_equity", 0.0)),
                   day_locked=bool(d.get("day_locked", False)))


def trend_weights(score: pd.Series, vol_daily: pd.Series, returns: pd.DataFrame,
                  p: PortfolioParams) -> dict[str, float]:
    """Risk-scaled trend weights for one rebalance. `returns` is the trailing hourly window."""
    s = score.dropna()
    s = s[s.abs() > p.min_score] if p.allow_shorts else s[s > p.min_score]
    v = vol_daily.reindex(s.index)
    s = s[np.isfinite(v) & (v > 0)]
    if s.empty:
        return {}
    w = s / vol_daily[s.index]
    for _ in range(2):  # size, drop dust positions, resize the survivors
        cov = returns[w.index].fillna(0.0).cov().to_numpy() * 24
        book_vol = float(np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 1e-12)))
        w = (w * (p.target_vol_daily / book_vol)).clip(-p.max_weight, p.max_weight)
        gross = w.abs().sum()
        if gross > p.gross_cap:
            w *= p.gross_cap / gross
        keep = w.abs() >= p.min_weight
        if keep.all() or not keep.any():
            break
        w = w[keep]
    return {k: float(x) for k, x in w.items() if abs(x) >= p.min_weight}


class Portfolio:
    def __init__(self, params: PortfolioParams, state: PortfolioState | None = None):
        self.p = params
        self.state = state or PortfolioState()

    def drawdown_scale(self, equity: float) -> float:
        s = self.state
        s.peak_equity = max(s.peak_equity, equity)
        dd = 1.0 - equity / s.peak_equity if s.peak_equity > 0 else 0.0
        scale = float(np.clip((self.p.dd_full - dd) / (self.p.dd_full - self.p.dd_start), 0.0, 1.0))
        step = self.p.brake_step
        return float(np.floor(scale / step + 1e-9) * step) if step > 0 else scale

    def rebalance(self, score: pd.Series, vol_daily: pd.Series, returns: pd.DataFrame,
                  prices: pd.Series) -> None:
        s = self.state
        s.base = trend_weights(score, vol_daily, returns, self.p)
        s.extreme = {k: s.extreme.get(k, float(prices[k])) for k in s.base}
        s.stopped = []

    def _apply_stops(self, prices: pd.Series, vol_daily: pd.Series) -> None:
        s, mult = self.state, self.p.stop_vol_mult
        if mult <= 0:
            return
        for pair, w in s.base.items():
            px = prices.get(pair, np.nan)
            if pair in s.stopped or not np.isfinite(px):
                continue
            dist = mult * float(vol_daily.get(pair, 0.05) or 0.05)
            if w > 0:
                s.extreme[pair] = max(s.extreme.get(pair, px), px)
                hit = px < s.extreme[pair] * (1 - dist)
            else:
                s.extreme[pair] = min(s.extreme.get(pair, px), px)
                hit = px > s.extreme[pair] * (1 + dist)
            if hit:
                s.stopped.append(pair)

    def targets(self, now: pd.Timestamp, score: pd.Series, vol_daily: pd.Series,
                returns: pd.DataFrame, prices: pd.Series, equity: float,
                force_rebalance: bool = False) -> dict[str, float]:
        """Advance one hour and return signed target weights."""
        if force_rebalance or now.hour == self.p.rebalance_hour_utc:
            self.rebalance(score, vol_daily, returns, prices)
        self._apply_stops(prices, vol_daily)
        scale = self.drawdown_scale(equity)
        if self._day_locked(now, equity):
            return {}
        return {k: w * scale for k, w in self.state.base.items() if k not in self.state.stopped}

    def _day_locked(self, now: pd.Timestamp, equity: float) -> bool:
        """Daily P&L lock: bank a good day or cap a bad one by going flat until the next anchor."""
        p, s = self.p, self.state
        if p.day_take_profit <= 0 and p.day_stop <= 0:
            return False
        key = (now - pd.Timedelta(hours=p.day_anchor_hour_utc)).strftime("%Y-%m-%d")
        if key != s.day_key:
            s.day_key, s.day_start_equity, s.day_locked = key, equity, False
        if not s.day_locked and s.day_start_equity > 0:
            day_ret = equity / s.day_start_equity - 1.0
            if (p.day_take_profit > 0 and day_ret >= p.day_take_profit) or (p.day_stop > 0 and day_ret <= -p.day_stop):
                s.day_locked = True
        return s.day_locked
