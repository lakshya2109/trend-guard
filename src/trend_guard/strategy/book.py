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

from trend_guard.config import BearModeConfig, BookConfig, CrashHedgeConfig, MomentumConfig, TrendConfig

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


def _side_split(vol: np.ndarray, power: float) -> np.ndarray:
    """Shares of one side's budget (sum 1): equal at power 0, else ~ 1/vol^power (calmer names get more)."""
    if power == 0.0:
        return np.full(len(vol), 1.0 / len(vol))
    inv = 1.0 / np.maximum(np.nan_to_num(vol, nan=0.05), 1e-4) ** power
    return inv / inv.sum()


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
    select = (f.index.hour - p.select_hour_utc) % p.select_every_h == 0  # 24: once a day
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
            w[longs] = (1 - p.short_share if shorts else 1.0) * _side_split(
                vol_d[t, longs], p.vol_weight_power
            )
        if shorts:
            w[shorts] = -p.short_share * _side_split(vol_d[t, shorts], p.vol_weight_power)
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
SLEEVES = ("v2_trend", "v2_momentum", "team_trend", "team_momentum", "jump_guard", "crash_hedge")


def bear_regime(close: pd.DataFrame, quote_volume: pd.DataFrame, p: BearModeConfig) -> np.ndarray:
    """(T,) bool: True while the market is in bear mode. Causal: hour t uses bars closed at or before t.

    Market = mean hourly log return of coins trading >= min_liquidity_usd a day (trailing 7-day mean).
    Enter when its lookback_h return falls below `enter`, leave when it is back above `exit`.
    """
    px = close.ffill(limit=3)
    liq = quote_volume.reindex_like(close).rolling(7 * H).sum() / 7 > p.min_liquidity_usd
    cum = np.log(px).diff().where(liq).mean(axis=1).fillna(0.0).cumsum().to_numpy()
    out = np.zeros(len(cum), dtype=bool)
    on = False
    for t in range(p.lookback_h, len(cum)):
        r = np.exp(cum[t] - cum[t - p.lookback_h]) - 1.0
        on = r < (p.exit if on else p.enter)
        out[t] = on
    return out


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
      + crash_hedge (short crypto-linked stock tokens against net long crypto while BTC is falling fast)
    """
    bm = cfg.bear_mode
    if bm.enabled:  # regime switch: the normal book, or v2 momentum alone (re-ranked faster) in bear mode
        base = cfg.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
        normal = book_components(close, quote_volume, base, allow_short, funding_z, oi_chg_z, ml_pred)
        bear_cfg = base.model_copy(
            update={
                "v2_share": 1.0,
                "team_share": 0.0,
                "trend_share": 0.0,
                "momentum_share": 1.0,
                "momentum": base.momentum.model_copy(update={"select_every_h": bm.select_every_h}),
            }
        )
        bear = book_components(close, quote_volume, bear_cfg, allow_short, funding_z, oi_chg_z, ml_pred)
        on = bear_regime(close, quote_volume, bm)[:, None]
        return {k: np.where(on, bear[k], normal[k]) for k in normal}
    for a, b, what in (
        (cfg.v2_share, cfg.team_share, "v2/team"),
        (cfg.trend_share, cfg.momentum_share, "v2 sleeves"),
        (cfg.team.trend_share, cfg.team.momentum_share, "team sleeves"),
    ):
        assert a + b <= 1.0 + 1e-9, f"{what} shares must sum to <= 1 (no leverage)"
    # the crash hedge's instruments are never part of the sleeves' universe
    hedge_cols = set(cfg.crash_hedge.instruments)
    full_close = close
    core = [c for c in close.columns if c not in hedge_cols]
    close, quote_volume = close[core], quote_volume.reindex(columns=close.columns)[core]
    zero = np.zeros(close.shape)
    comp = {k: zero.copy() for k in SLEEVES if k != "crash_hedge"}
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
    comp = {k: v * scale for k, v in comp.items()}
    # expand back to the full panel and add the crash hedge on the book as it stands
    pos = [full_close.columns.get_loc(c) for c in close.columns]
    out = {}
    for k, v in comp.items():
        w = np.zeros(full_close.shape)
        w[:, pos] = v
        out[k] = w
    if cfg.jump_guard.enabled:  # an adjustment component, so the components still sum to the book
        total = sum(out.values())
        out["jump_guard"] = (
            jump_guard(full_close, total, cfg.jump_guard.threshold, cfg.jump_guard.hold_h) - total
        )
    else:
        out["jump_guard"] = np.zeros(full_close.shape)
    out["crash_hedge"] = np.zeros(full_close.shape)
    h = cfg.crash_hedge
    if h.enabled and allow_short:
        out["crash_hedge"] = crash_hedge(full_close, sum(out.values()), h)
        gross = np.abs(sum(out.values())).sum(axis=1, keepdims=True)
        fix = np.where(gross > cfg.max_gross, cfg.max_gross / np.maximum(gross, 1e-12), 1.0)
        out = {k: v * fix for k, v in out.items()}
    return out


def jump_guard(close: pd.DataFrame, book: np.ndarray, threshold: float, hold_h: int) -> np.ndarray:
    """The book with new or larger longs blocked for hold_h hours after a 1-hour rise above threshold.

    Causal: the bar's return is known at its close, when the hour's targets are set. A blocked long keeps
    the previous hour's (guarded) weight: never opened, never added to; reductions, closes and shorts pass.
    """
    lp = np.log(close.ffill(limit=3).to_numpy())
    r1 = np.vstack([np.zeros((1, lp.shape[1])), np.nan_to_num(np.diff(lp, axis=0))])
    jumped = r1 > np.log1p(threshold)
    since = np.full(book.shape[1], np.iinfo(np.int64).max // 2)
    out = np.nan_to_num(book).copy()
    prev = np.zeros(book.shape[1])
    for t in range(book.shape[0]):
        since = np.where(jumped[t], 0, since + 1)
        guarded = since <= hold_h
        w = out[t]
        block_open = guarded & (w > 0) & (prev <= 0)
        block_add = guarded & (w > prev) & (prev > 0)
        w[block_open] = 0.0
        w[block_add] = prev[block_add]
        prev = w
    return out


def crash_hedge(close: pd.DataFrame, book: np.ndarray, h: CrashHedgeConfig) -> np.ndarray:
    """Short the instruments, worth h.size x the book's net long crypto, while the reference (BTC) is
    falling fast: 72h return below trigger_72h or 24h return below trigger_24h. The state is set at the
    check hours and held until the next check; it switches off as soon as a check finds no crash.
    Instruments without a price at an hour get no weight (the others share the hedge)."""
    W = np.zeros(close.shape)
    inst = [c for c in h.instruments if c in close.columns]
    if not inst or h.reference not in close.columns:
        return W
    ref = close[h.reference].ffill(limit=3)
    raw = ((ref / ref.shift(72) - 1) < h.trigger_72h) | ((ref / ref.shift(24) - 1) < h.trigger_24h)
    check = np.isin(close.index.hour, h.check_hours_utc)
    state = (
        pd.Series(np.where(check, raw.to_numpy(dtype=float), np.nan), index=close.index)
        .ffill()
        .fillna(0.0)
        .to_numpy()
    )
    idx = [close.columns.get_loc(c) for c in inst]
    live = np.isfinite(close.iloc[:, idx].ffill(limit=3).to_numpy())
    n = np.maximum(live.sum(axis=1), 1)
    net_long = np.clip(np.nansum(np.delete(book, idx, axis=1), axis=1), 0.0, None)
    for j, col in enumerate(idx):
        W[:, col] = -h.size * state * net_long * live[:, j] / n
    return W


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
