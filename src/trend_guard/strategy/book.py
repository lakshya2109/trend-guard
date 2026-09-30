"""The trading book: a low-volatility trend sleeve plus a long/short cross-sectional momentum sleeve.

Pure functions of an hourly market panel (no network, no disk, no clock). `book_weights` returns
target weights (fractions of equity, shorts negative) for every hour of the panel; the backtest reads
every row, the live engine reads the last one, so what is tested is exactly what trades.

  trend sleeve     long-only time-series trend: score = mean over lookbacks of the log return
                   divided by the volatility expected over that horizon (a t-statistic), clipped;
                   weight ~ score / vol for names above a threshold, the book scaled to a low daily
                   volatility target with the trailing covariance; rebalanced once a day. Four
                   members (two lookback sets x two thresholds) are averaged.
  momentum sleeve  cross-sectional: once a day rank liquid coins by risk-adjusted momentum, long
                   the best k, short the worst k (a holding is kept while it stays within `buffer`
                   ranks), equal weight per side, resized every hour to a daily volatility target
                   with a constant-correlation risk model. Four members (two lookback sets x two
                   skips) are averaged.

Why ensembles: single parameter choices are noisy at a 14-day horizon (see docs/STRATEGY.md);
averaging members keeps the signal and cancels part of the selection noise and the turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_guard.config import BookConfig, MomentumConfig, TrendConfig

H = 24  # hours per day


@dataclass
class Features:
    """Aligned (T, N) arrays for one sleeve. Every value at row t uses bars closed at or before t."""

    close: np.ndarray  # forward-filled at most 3 bars
    logp: np.ndarray
    ret: np.ndarray  # hourly log returns (NaN where missing)
    vol_h: np.ndarray  # hourly volatility estimate
    liq: np.ndarray  # trailing 7-day mean daily quote volume (USD)
    age: np.ndarray  # number of bars seen so far
    index: pd.DatetimeIndex
    cols: list[str]


def features(close: pd.DataFrame, quote_volume: pd.DataFrame, vol_kind: str, vol_param: float) -> Features:
    """vol_kind 'ewm': RiskMetrics EWMA with decay vol_param; 'rolling': rolling std over vol_param hours."""
    px = close.ffill(limit=3)
    logp = np.log(px)
    ret = logp.diff()
    if vol_kind == "ewm":
        vol = np.sqrt((ret**2).ewm(alpha=1 - vol_param, min_periods=28).mean())
    else:
        win = int(vol_param)
        vol = ret.rolling(win, min_periods=win // 2).std()
    liq = quote_volume.reindex_like(close).rolling(7 * H, min_periods=3 * H).sum() / 7
    age = close.notna().cumsum()
    return Features(
        px.to_numpy(),
        logp.to_numpy(),
        ret.to_numpy(),
        vol.to_numpy(),
        liq.to_numpy(),
        age.to_numpy(),
        close.index,
        list(close.columns),
    )


def tstat_score(f: Features, lookbacks: list[int], skip: int = 0) -> np.ndarray:
    """Mean over lookbacks L of (log return from t-skip-L to t-skip) / (vol_h * sqrt(L))."""
    T = f.logp.shape[0]
    out = np.zeros_like(f.logp)
    for L in lookbacks:
        a = np.full_like(f.logp, np.nan)
        if skip + L < T:
            a[skip + L :] = f.logp[L : T - skip] - f.logp[: T - skip - L]
        out = out + a / (f.vol_h * np.sqrt(L))
    return out / len(lookbacks)


def _cov_scale(w: np.ndarray, rets: np.ndarray, target_daily: float, gross_cap: float) -> float:
    """Factor that brings the book's daily vol (sample covariance of hourly returns x 24) to target."""
    held = w != 0
    if not held.any():
        return 0.0
    R = np.nan_to_num(rets[:, held])
    C = np.atleast_2d(np.cov(R, rowvar=False)) * H
    ww = w[held]
    vol = float(np.sqrt(max(ww @ C @ ww, 1e-12)))
    return min(target_daily / vol, gross_cap / np.abs(ww).sum())


def _cc_scale(w: np.ndarray, vol_d: np.ndarray, target_daily: float, gross_cap: float, rho: float) -> float:
    """Same, under a constant-correlation model (robust with few names and short windows)."""
    held = w != 0
    if not held.any():
        return 0.0
    a = w[held] * np.nan_to_num(vol_d[held], nan=0.05)
    var = (a**2).sum() + rho * (a.sum() ** 2 - (a**2).sum())
    return min(target_daily / np.sqrt(max(var, 1e-12)), gross_cap / np.abs(w[held]).sum())


# ----------------------------------------------------------------------------- trend sleeve
def trend_member(f: Features, p: TrendConfig, lookbacks: list[int], min_score: float) -> np.ndarray:
    T, N = f.close.shape
    sc = np.clip(tstat_score(f, lookbacks), -p.score_cap, p.score_cap) / p.score_cap
    vol_d = f.vol_h * np.sqrt(H)
    excluded = np.array([c in p.exclude for c in f.cols])
    elig = (
        (f.liq >= p.min_liquidity_usd)
        & (f.age >= p.min_age_h)
        & np.isfinite(sc)
        & np.isfinite(vol_d)
        & (vol_d > 0)
        & ~excluded
    )
    decide = f.index.hour == p.rebalance_hour_utc
    W = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        if decide[t]:
            s = np.where(elig[t], sc[t], 0.0)
            pick = s > min_score  # long-only
            w = np.zeros(N)
            w[pick] = s[pick] / vol_d[t, pick]
            window = f.ret[max(0, t - p.cov_hours + 1) : t + 1]
            for _ in range(2):  # size, drop positions too small to pay their costs, resize survivors
                if not (w != 0).any():
                    break
                w = np.clip(
                    w * _cov_scale(w, window, p.target_vol_daily, p.gross_cap), -p.max_weight, p.max_weight
                )
                gross = np.abs(w).sum()
                if gross > p.gross_cap:
                    w *= p.gross_cap / gross
                small = (w != 0) & (np.abs(w) < p.min_weight)
                if not small.any():
                    break
                w[small] = 0.0
            w[np.abs(w) < p.min_weight] = 0.0
            cur = w
        W[t] = cur
    return W


def trend_sleeve(
    close: pd.DataFrame, quote_volume: pd.DataFrame, p: TrendConfig, min_weights: list[float] | None = None
) -> np.ndarray:
    """Average of members over lookback_sets x min_scores (x min_weights, for research sweeps)."""
    f = features(close, quote_volume, "rolling", p.vol_window_h)
    floors = min_weights or [p.min_weight]
    members = [
        trend_member(f, p.model_copy(update={"min_weight": mw}), lb, th)
        for lb in p.lookback_sets
        for th in p.min_scores
        for mw in floors
    ]
    return np.mean(members, axis=0)


# ----------------------------------------------------------------------------- momentum sleeve
def _pick(order_value: np.ndarray, ok: np.ndarray, prev: list[int], k: int, buffer: int) -> list[int]:
    """Top-k of order_value among ok names; previous holdings survive while ranked within buffer."""
    n = len(order_value)
    ranked = np.argsort(-np.where(ok, order_value, -np.inf))
    rank = np.empty(n, dtype=int)
    rank[ranked] = np.arange(1, n + 1)
    keep = [i for i in prev if ok[i] and rank[i] <= buffer]
    fresh = [int(i) for i in ranked if ok[i] and i not in keep]
    return (keep + fresh[: max(0, k - len(keep))])[:k]


def momentum_member(
    f: Features, p: MomentumConfig, lookbacks: list[int], skip: int, allow_short: bool
) -> np.ndarray:
    T, N = f.close.shape
    sc = tstat_score(f, lookbacks, skip)
    vol_d = f.vol_h * np.sqrt(H)
    excluded = np.array([c in p.exclude for c in f.cols])
    elig = (
        (f.liq >= p.min_liquidity_usd)
        & (f.age >= p.min_age_h)
        & np.isfinite(sc)
        & ~excluded
        & np.isfinite(f.close)
    )
    select = f.index.hour == p.select_hour_utc
    W = np.zeros((T, N))
    longs: list[int] = []
    shorts: list[int] = []
    for t in range(T):
        if select[t]:
            s = sc[t]
            longs = _pick(s, elig[t] & (s > 0), longs, p.k, p.buffer)
            shorts = _pick(-s, elig[t] & (s < 0), shorts, p.k, p.buffer) if allow_short else []
        if not longs and not shorts:
            continue
        w = np.zeros(N)
        if longs:
            w[longs] = (1 - p.short_share if shorts else 1.0) / len(longs)
        if shorts:
            w[shorts] = -p.short_share / len(shorts)
        W[t] = w * _cc_scale(w, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)  # resized hourly
    return W


def momentum_sleeve(
    close: pd.DataFrame, quote_volume: pd.DataFrame, p: MomentumConfig, allow_short: bool = True
) -> np.ndarray:
    f = features(close, quote_volume, "ewm", p.ewma_lambda)
    members = [momentum_member(f, p, lb, sk, allow_short) for lb in p.lookback_sets for sk in p.skips]
    return np.mean(members, axis=0)


# ----------------------------------------------------------------------------- the book
def book_weights(
    close: pd.DataFrame, quote_volume: pd.DataFrame, cfg: BookConfig, allow_short: bool = True
) -> pd.DataFrame:
    """Account-level target weights for every hour: capital-weighted sum of both sleeves, netted per pair.

    Longs + short collateral stay within 1x equity: each sleeve caps its own gross, and the shares sum
    to at most 1 (asserted), so the combined gross can never exceed 1.
    """
    assert cfg.trend_share + cfg.momentum_share <= 1.0 + 1e-9, "sleeve shares must sum to <= 1 (no leverage)"
    W = np.zeros(close.shape)
    if cfg.trend_share > 0:
        W += cfg.trend_share * trend_sleeve(close, quote_volume, cfg.trend)
    if cfg.momentum_share > 0:
        W += cfg.momentum_share * momentum_sleeve(close, quote_volume, cfg.momentum, allow_short)
    return pd.DataFrame(W, index=close.index, columns=close.columns)
