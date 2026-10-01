"""Contest replay: every 14-day window, scored the way the hackathon scores.

Protocol (shared with the team harness so results are directly comparable):
  - a window starts every day at `window_hour_utc` (12:00, the contest start hour) from
    `first_window` to `last_window`, flat in $100k cash, and runs 14 days;
  - each hour the strategy's target weights are traded at that hour's close (the live bot trades
    3 minutes after the candle closes), paying fee + slippage per side on every fill;
  - trade rule: act at decision hours, whenever a target changes, and at the window open; skip a
    resize smaller than max(min_trade_weight, min_trade_frac x |target|) unless it opens or closes;
  - the catastrophe breaker multiplies every target once the window's drawdown reaches dd_level;
  - score = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar on daily marks from the open, non-annualised,
    zero-centred downside deviation, Calmar = mean daily return / max drawdown.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from trend_guard.config import BacktestConfig, BreakerConfig, ExecutionConfig

EPS = 1e-8


def window_starts(index: pd.DatetimeIndex, bt: BacktestConfig) -> np.ndarray:
    """Positions in `index` where a full window can start."""
    hours = bt.window_days * 24
    out = []
    for s in pd.date_range(
        f"{bt.first_window} {bt.window_hour_utc:02d}:00",
        f"{bt.last_window} {bt.window_hour_utc:02d}:00",
        freq="1D",
        tz="UTC",
    ):
        if s in index:
            i = index.get_loc(s)
            if i + hours < len(index):
                out.append(i)
    return np.array(out, dtype=np.int64)


@nb.njit(cache=True)
def _simulate(W, px, act, starts, H, cost, min_w, min_frac, capital, dd_lvl, dd_mult, dd_rec):
    nwin = starts.shape[0]
    N = W.shape[1]
    equity = np.empty((nwin, H + 1))
    turnover = np.zeros(nwin)
    traded = np.zeros((nwin, H))  # notional traded per hour (x capital): active days come from this
    for k in range(nwin):
        st = starts[k]
        qty = np.zeros(N)
        prev = np.zeros(N)
        cash = capital
        peak = capital
        brake = 1.0
        for s in range(H + 1):
            t = st + s
            e = cash
            for i in range(N):
                p = px[t, i]
                if p > 0.0 and np.isfinite(p):
                    e += qty[i] * p
            equity[k, s] = e
            if s == H:
                break
            if e > peak:
                peak = e
            dd = 1.0 - e / peak
            if dd >= dd_lvl:
                brake = dd_mult
            elif dd <= dd_rec:
                brake = 1.0
            for i in range(N):
                p = px[t, i]
                ok = p > 0.0 and np.isfinite(p)
                tgt = W[t, i] * brake if ok else 0.0
                if not np.isfinite(tgt):
                    tgt = 0.0
                cur = qty[i] * p / e if ok else 0.0
                changed = abs(tgt - prev[i]) > 1e-9
                prev[i] = tgt
                if not ((changed or s == 0 or act[t]) and ok and abs(tgt - cur) > 1e-9):
                    continue
                delta = tgt - cur
                if tgt != 0.0 and cur != 0.0 and abs(delta) < max(min_w, min_frac * abs(tgt)):
                    continue  # resize too small to pay for itself
                if tgt == 0.0:
                    notional = -qty[i] * p
                    qty[i] = 0.0
                else:
                    notional = delta * e
                    qty[i] += notional / p
                cash -= notional + abs(notional) * cost
                turnover[k] += abs(notional) / capital
                traded[k, s] += abs(notional) / capital
    return equity, turnover, traded


def simulate(
    W: np.ndarray,
    close: pd.DataFrame,
    starts: np.ndarray,
    bt: BacktestConfig,
    ex: ExecutionConfig,
    breaker: BreakerConfig,
    with_trades: bool = False,
) -> tuple[np.ndarray, ...]:
    """(equity paths (n_windows, hours+1), turnover per window as a multiple of capital), plus the
    notional traded per hour (n_windows, hours) when with_trades is set."""
    px = close.ffill(limit=3).to_numpy(dtype=np.float64)
    act = np.isin(close.index.hour, ex.decision_hours_utc)
    eq, turnover, traded = _simulate(
        np.ascontiguousarray(W, dtype=np.float64),
        px,
        act,
        starts,
        bt.window_days * 24,
        bt.fee + bt.slippage,
        ex.min_trade_weight,
        ex.min_trade_frac,
        bt.capital,
        breaker.dd_level,
        breaker.dd_mult,
        breaker.dd_recover,
    )
    return (eq, turnover, traded) if with_trades else (eq, turnover)


def contest_score(equity: np.ndarray) -> np.ndarray:
    """Composite per window from hourly equity paths (rows = windows)."""
    marks = equity[:, ::24]
    r = marks[:, 1:] / marks[:, :-1] - 1.0
    mean = r.mean(axis=1)
    mdd = np.max(1.0 - marks / np.maximum.accumulate(marks, axis=1), axis=1)
    down = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=1))
    return (
        0.4 * mean / np.maximum(down, EPS)
        + 0.3 * mean / np.maximum(r.std(axis=1), EPS)
        + 0.3 * mean / np.maximum(mdd, EPS)
    )


def summarize(
    equity: np.ndarray,
    turnover: np.ndarray,
    start_times: pd.DatetimeIndex,
    periods: dict[str, tuple[str | None, str | None]],
) -> pd.DataFrame:
    """The team's comparison table: median score, median return, share above -1%, share positive,
    worst window, plus tail and cost columns, for each period (by window start date)."""
    ret = equity[:, -1] / equity[:, 0] - 1.0
    score = contest_score(equity)
    mdd = np.max(1.0 - equity / np.maximum.accumulate(equity, axis=1), axis=1)
    rows = {}
    for name, (lo, hi) in periods.items():
        m = np.ones(len(ret), dtype=bool)
        if lo:
            m &= np.asarray(start_times >= pd.Timestamp(lo, tz="UTC"))
        if hi:
            m &= np.asarray(start_times < pd.Timestamp(hi, tz="UTC"))
        if not m.any():
            continue
        rows[name] = {
            "score": float(np.median(score[m])),
            "median_return": float(np.median(ret[m])),
            "above_-1%": float(np.mean(ret[m] > -0.01)),
            "positive": float(np.mean(ret[m] > 0)),
            "worst": float(ret[m].min()),
            "p10_return": float(np.quantile(ret[m], 0.1)),
            "mean_return": float(ret[m].mean()),
            "p95_drawdown": float(np.quantile(mdd[m], 0.95)),
            "turnover": float(np.median(turnover[m])),
            "windows": int(m.sum()),
        }
    return pd.DataFrame(rows).T


def competition_metrics(
    equity: np.ndarray, traded: np.ndarray, start_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-window metrics as the previous edition's leaderboard reported them (backtest/benchmark.py):
    Sharpe and Sortino on daily returns (not annualised), Calmar = total return / max drawdown, and
    active days = UTC calendar dates with at least one trade (a 12:00 -> 12:00 fortnight spans 15)."""
    marks = equity[:, ::24]
    r = marks[:, 1:] / marks[:, :-1] - 1.0
    mean = r.mean(axis=1)
    ret = equity[:, -1] / equity[:, 0] - 1.0
    mdd = np.max(1.0 - equity / np.maximum.accumulate(equity, axis=1), axis=1)
    down = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=1))
    active = np.zeros(len(start_times), dtype=int)
    for k, st in enumerate(start_times):
        hours = st + pd.to_timedelta(
            np.arange(1, traded.shape[1] + 1), unit="h"
        )  # trades go out after each bar
        active[k] = len(set(hours[traded[k] > 0].date))
    return pd.DataFrame(
        {
            "return": ret,
            "sharpe": mean / np.maximum(r.std(axis=1, ddof=1), EPS),
            "sortino": mean / np.maximum(down, EPS),
            "calmar_total": ret / np.maximum(mdd, EPS),
            "max_drawdown": mdd,
            "active_days": active,
        },
        index=start_times,
    )


def non_overlapping(start_times: pd.DatetimeIndex, days: int, offset: int = 0) -> np.ndarray:
    """Positions of windows that share no day: every `days`-th daily start, beginning at `offset`."""
    return np.arange(offset, len(start_times), days)
