"""Adapter: the teammates' combined book as an hourly target-weight matrix.

Runs their vendored code (strategy/team/, unmodified) over the price panel the way their own
backtest does:
  - swyam-rd's momentum book: compute_signals + run_model, with the vol target overridden to the
    1.5%/day their combined config uses;
  - Gunek-945's trend book: the stateful Portfolio replayed hour by hour, rebalancing at its own
    12:00 UTC hour. Equity is passed as a constant so the book's internal drawdown brake stays idle;
    the account-level breaker (risk/breaker.py) covers drawdowns for the whole account instead.
The split between the two (70% trend / 30% momentum) comes from config.book.team.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from trend_guard.config import TeamConfig
from trend_guard.strategy.team.momentum_model import ModelParams, compute_signals, run_model
from trend_guard.strategy.team.trend_portfolio import Portfolio, PortfolioParams
from trend_guard.strategy.team.trend_signal import SignalParams, compute_features

_CONSTANT_EQUITY = 100_000.0  # keeps the trend book's internal drawdown brake idle (see docstring)


def team_momentum(
    close: pd.DataFrame, quote_volume: pd.DataFrame, vol_target: float, allow_short: bool
) -> np.ndarray:
    mp = replace(ModelParams(), vol_target_daily=vol_target, allow_short=allow_short)
    sig = compute_signals({"close": close, "quote_volume": quote_volume}, mp)
    Wc = run_model(sig, mp)
    W = np.zeros(close.shape)
    for j, pair in enumerate(sig.pairs):
        W[:, close.columns.get_loc(pair)] = Wc[:, j]
    return W


def team_trend(close: pd.DataFrame, quote_volume: pd.DataFrame) -> np.ndarray:
    feats = compute_features(close, quote_volume, SignalParams())
    book = Portfolio(PortfolioParams())
    px = close.ffill(limit=3)
    col = {p: i for i, p in enumerate(close.columns)}
    W = np.zeros(close.shape)
    hours = np.flatnonzero(close.index.hour == book.p.rebalance_hour_utc)
    if not len(hours):
        return W
    first = int(hours[0])  # start at the book's first rebalance hour, as its own backtest does
    for pos in range(first, len(close)):
        window = feats.returns.iloc[max(0, pos - book.p.cov_hours + 1) : pos + 1]
        targets = book.targets(
            close.index[pos],
            feats.score.iloc[pos],
            feats.vol_daily.iloc[pos],
            window,
            px.iloc[pos],
            _CONSTANT_EQUITY,
            force_rebalance=pos == first,
        )
        for pair, w in targets.items():
            W[pos, col[pair]] = w
    return W


def team_weights(
    close: pd.DataFrame, quote_volume: pd.DataFrame, cfg: TeamConfig, allow_short: bool = True
) -> np.ndarray:
    """cfg.trend_share x trend book + cfg.momentum_share x momentum book (fractions of equity)."""
    W = np.zeros(close.shape)
    if cfg.trend_share > 0:
        W += cfg.trend_share * team_trend(close, quote_volume)
    if cfg.momentum_share > 0:
        W += cfg.momentum_share * team_momentum(close, quote_volume, cfg.momentum_vol_target, allow_short)
    return W
