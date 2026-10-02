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
                   ranks; no new shorts on coins already down > 40% in 60 days), equal weight
                   per side, resized every hour to a daily volatility target with a
                   constant-correlation risk model. Four members (two lookback sets x two skips)
                   are averaged.

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


def _weighted_cov(R: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reliability-weighted sample covariance (bias-corrected): weights sum to 1, more recent
    rows (last row = most recent) can carry more weight than older ones."""
    mean = (weights[:, None] * R).sum(axis=0)
    Rc = R - mean
    correction = max(1.0 - (weights**2).sum(), 1e-12)
    return ((Rc * weights[:, None]).T @ Rc) / correction


def _cov_scale(
    w: np.ndarray,
    rets: np.ndarray,
    target_daily: float,
    gross_cap: float,
    shrink: float = 0.0,
    rho: float = 0.7,
    ewma_halflife: float = 0.0,
) -> float:
    """Factor that brings the book's daily vol (sample covariance of hourly returns x 24) to
    target. `_cc_scale`'s own docstring calls the constant-correlation model "robust with few
    names and short windows" -- exactly the regime the trend sleeve runs in (up to several dozen
    held names, 168 hourly observations: the sample covariance matrix can be close to singular).
    `shrink` (0 = pure sample, as before; 1 = pure constant-correlation) linearly blends the
    sample covariance with a constant-correlation target at the SAME variances -- a direct,
    data-light version of Ledoit-Wolf shrinkage, reusing the momentum sleeve's existing
    constant-correlation code as the shrinkage target rather than a new model.

    `ewma_halflife` (hours, 0 = disabled -> equal-weighted, as before): makes the covariance
    recency-weighted within the same `rets` window, so a longer window (more data -> a more
    stable correlation structure) doesn't also mean slower to react when today's volatility is
    genuinely higher than the window's average -- the same idea the momentum sleeve's own vol
    estimate already uses (`ewma_lambda`), just not previously applied to this covariance matrix.
    """
    held = w != 0
    if not held.any():
        return 0.0
    R = np.nan_to_num(rets[:, held])
    if ewma_halflife > 0.0:
        n = R.shape[0]
        decay = 0.5 ** (1.0 / ewma_halflife)
        raw = decay ** np.arange(n - 1, -1, -1)
        weights = raw / raw.sum()
        C = _weighted_cov(R, weights) * H
    else:
        C = np.atleast_2d(np.cov(R, rowvar=False)) * H
    if shrink > 0.0:
        std = np.sqrt(np.clip(np.diag(C), 0.0, None))
        target = rho * np.outer(std, std)
        np.fill_diagonal(target, np.diag(C))
        C = (1.0 - shrink) * C + shrink * target
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
                    w
                    * _cov_scale(
                        w,
                        window,
                        p.target_vol_daily,
                        p.gross_cap,
                        p.cov_shrink,
                        p.cov_shrink_rho,
                        p.cov_ewma_halflife_h,
                    ),
                    -p.max_weight,
                    p.max_weight,
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
    f: Features,
    p: MomentumConfig,
    lookbacks: list[int],
    skip: int,
    allow_short: bool,
    funding_z: np.ndarray | None = None,
    oi_chg_z: np.ndarray | None = None,
    ml_pred: np.ndarray | None = None,
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
    # crash filter: a coin already down more than short_crash_return over the lookback is not a new short
    crashed = np.zeros((T, N), dtype=bool)
    if p.short_crash_return is not None:
        L = p.short_crash_lookback_h
        if L < T:
            drop = f.logp[L:] - f.logp[:-L]
            crashed[L:] = np.nan_to_num(drop, nan=0.0) < np.log1p(p.short_crash_return)
    # Derivatives crowding filter (config docstring for the exact rule): a continuous, every-hour
    # multiplier, the same shape as the other overlays in book_components -- computed once here,
    # outside the decision loop, since it doesn't depend on which names are currently held.
    crowd_on = p.crowd_enabled and funding_z is not None and oi_chg_z is not None
    if crowd_on:
        long_squeeze_down = (
            np.isfinite(funding_z)
            & np.isfinite(oi_chg_z)
            & (funding_z > p.crowd_funding_z)
            & (oi_chg_z > p.crowd_oi_change_z)
            & (sc > p.crowd_price_extended)
        )
        short_squeeze_up = np.isfinite(funding_z) & (funding_z < -p.crowd_funding_z)
    # Cross-sectional ML filter (config docstring for the exact rule): a qualifying condition at
    # selection time, the same shape as the crash filter -- an already-held name is never forced
    # out just because the model's prediction moved against it, only a NEW entry is blocked.
    # A missing prediction (NaN -- no walk-forward model fit yet, ~180 days of warmup at the start
    # of the backtest) must mean "abstain", not "reject": the first version of this filter treated
    # NaN as a hard no, which blocked every new long for the model's entire warmup period and
    # badly distorted the 2024-25 comparison (bug caught by re-reading this code, not a test --
    # the existing tests all used `ml_pred` arrays with no NaNs in them).
    ml_on = p.ml_enabled and ml_pred is not None
    W = np.zeros((T, N))
    longs: list[int] = []
    shorts: list[int] = []
    for t in range(T):
        if select[t]:
            s = sc[t]
            held_long = np.zeros(N, dtype=bool)
            held_long[longs] = True
            long_elig = elig[t] & (s > 0)
            if ml_on:
                pred_t = ml_pred[t]
                ml_ok_long = ~np.isfinite(pred_t) | (pred_t > p.ml_min_pred)
                long_elig = long_elig & (held_long | ml_ok_long)
            longs = _pick(s, long_elig, longs, p.k, p.buffer)
            held = np.zeros(N, dtype=bool)
            held[shorts] = True
            ok_short = elig[t] & (s < 0) & (~crashed[t] | held)  # held shorts are not forced out
            if ml_on:
                pred_t = ml_pred[t]
                ml_ok_short = ~np.isfinite(pred_t) | (pred_t < -p.ml_min_pred)
                ok_short = ok_short & (held | ml_ok_short)
            shorts = _pick(-s, ok_short, shorts, p.k, p.buffer) if allow_short else []
        if not longs and not shorts:
            continue
        w = np.zeros(N)
        if longs:
            w[longs] = (1 - p.short_share if shorts else 1.0) / len(longs)
        if shorts:
            w[shorts] = -p.short_share / len(shorts)
        scaled = w * _cc_scale(w, vol_d[t], p.target_vol_daily, p.gross_cap, p.avg_corr)  # resized hourly
        if crowd_on:
            # Applied AFTER sizing, like BUILD_SPEC's other overlays (w = w * crowd_mult * ...,
            # Section 5.1) -- applying it before the constant-correlation rescale would let the
            # rescale partly or fully undo the dampening (it targets a fixed portfolio vol, so
            # shrinking part of the book pre-scale just makes the rescale factor bigger).
            crowded = np.zeros(N, dtype=bool)
            crowded[longs] = long_squeeze_down[t, longs] if longs else False
            crowded[shorts] |= short_squeeze_up[t, shorts] if shorts else False
            scaled = np.where(crowded, scaled * p.crowd_mult, scaled)
        W[t] = scaled
    return W


def momentum_sleeve(
    close: pd.DataFrame,
    quote_volume: pd.DataFrame,
    p: MomentumConfig,
    allow_short: bool = True,
    funding_z: np.ndarray | None = None,
    oi_chg_z: np.ndarray | None = None,
    ml_pred: np.ndarray | None = None,
) -> np.ndarray:
    f = features(close, quote_volume, "ewm", p.ewma_lambda)
    members = [
        momentum_member(f, p, lb, sk, allow_short, funding_z, oi_chg_z, ml_pred)
        for lb in p.lookback_sets
        for sk in p.skips
    ]
    return np.mean(members, axis=0)


# ----------------------------------------------------------------------------- the book
SLEEVES = ("v2_trend", "v2_momentum", "team_trend", "team_momentum")


def book_components(
    close: pd.DataFrame,
    quote_volume: pd.DataFrame,
    cfg: BookConfig,
    allow_short: bool = True,
    funding_z: np.ndarray | None = None,
    oi_chg_z: np.ndarray | None = None,
    ml_pred: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Each sleeve's contribution to the account target (T, N), already multiplied by its capital share,
    the risk multiplier and the hour's gross cap, so the components sum exactly to book_weights.

        v2_share x (trend_share x trend sleeve + momentum_share x momentum sleeve)
      + team_share x (team trend_share x Gunek-945 trend book + team momentum_share x swyam-rd momentum book)
    """
    for a, b, what in (
        (cfg.v2_share, cfg.team_share, "v2/team"),
        (cfg.trend_share, cfg.momentum_share, "v2 sleeves"),
        (cfg.team.trend_share, cfg.team.momentum_share, "team sleeves"),
    ):
        assert a + b <= 1.0 + 1e-9, f"{what} shares must sum to <= 1 (no leverage)"
    zero = np.zeros(close.shape)
    comp = {k: zero.copy() for k in SLEEVES}
    if cfg.v2_share > 0 and cfg.trend_share > 0:
        comp["v2_trend"] = cfg.v2_share * cfg.trend_share * trend_sleeve(close, quote_volume, cfg.trend)
    if cfg.v2_share > 0 and cfg.momentum_share > 0:
        comp["v2_momentum"] = (
            cfg.v2_share
            * cfg.momentum_share
            * momentum_sleeve(close, quote_volume, cfg.momentum, allow_short, funding_z, oi_chg_z, ml_pred)
        )
    if cfg.team_share > 0:
        from trend_guard.strategy.team_book import team_momentum, team_trend  # teammates' code

        t = cfg.team
        if t.trend_share > 0:
            comp["team_trend"] = cfg.team_share * t.trend_share * team_trend(close, quote_volume)
        if t.momentum_share > 0:
            comp["team_momentum"] = (
                cfg.team_share
                * t.momentum_share
                * team_momentum(close, quote_volume, t.momentum_vol_target, allow_short)
            )
    # risk dial: scale the whole book, then cap gross per hour so it can never exceed max_gross (no leverage)
    total = sum(comp.values()) * cfg.risk_multiplier
    gross = np.abs(total).sum(axis=1, keepdims=True)
    scale = cfg.risk_multiplier * np.where(
        gross > cfg.max_gross, cfg.max_gross / np.maximum(gross, 1e-12), 1.0
    )
    return {k: v * scale for k, v in comp.items()}


def book_weights(
    close: pd.DataFrame,
    quote_volume: pd.DataFrame,
    cfg: BookConfig,
    allow_short: bool = True,
    funding_z: np.ndarray | None = None,
    oi_chg_z: np.ndarray | None = None,
    ml_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    """Account-level target weights for every hour (the sum of book_components), netted per pair.

    The book is multiplied by risk_multiplier and then capped hour by hour at max_gross (<= 1), so the
    account's long notional plus short collateral can never exceed equity.
    """
    comp = book_components(close, quote_volume, cfg, allow_short, funding_z, oi_chg_z, ml_pred)
    W = sum(comp.values())
    return pd.DataFrame(W, index=close.index, columns=close.columns)
