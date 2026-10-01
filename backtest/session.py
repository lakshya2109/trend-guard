"""A simulated trading session: run the live book over any [start, end) window and log every trade.

No data leakage, by construction and by check:
  1. the price panel handed to the strategy stops at `end`: no bar after the session exists in memory;
  2. history before `start` is limited to the live engine's own `history_hours` (what the live bot
     would have loaded), so the session sees exactly what the live bot would have seen;
  3. `leakage_check` recomputes every target with the second half of the session deleted and asserts
     the first half is unchanged, i.e. no target ever depended on a later bar.

Execution follows backtest/contest.py (the scored rule, identical to the live plan_orders): each hour's
targets are traded at that hour's close plus fee + slippage per side; act at decision hours, on target
changes, and at the open; skip resizes smaller than max(min_trade_weight, min_trade_frac x |target|)
unless opening or closing; the catastrophe breaker scales targets once the drawdown reaches dd_level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.benchmark import WINNER, rank_vs_field
from trend_guard.config import StrategyConfig
from trend_guard.strategy.book import SLEEVES, book_components

HOUR = pd.Timedelta(hours=1)


@dataclass
class Trade:
    time: pd.Timestamp  # when the order is sent: just after the decision candle closes
    pair: str
    # BUY / SELL move a long up / down; SHORT / COVER open-add / reduce-close a short
    action: str
    reason: str  # open / add / reduce / close / flip
    qty: float
    price: float
    notional: float
    fee: float
    weight_before: float
    weight_after: float
    drivers: dict[str, float] = field(default_factory=dict)  # each sleeve's share of the new target


@dataclass
class SessionResult:
    equity: pd.Series  # hourly marks from the start (cash-only) to the end
    trades: list[Trade]
    exposure: pd.DataFrame  # hourly gross / net / long / short as fractions of equity
    pnl_by_pair: pd.Series
    leakage_ok: bool
    leakage_detail: str


def _classify(before: float, after: float) -> list[tuple[str, str, float, float]]:
    """Split a weight change into exchange actions: (action, reason, from, to). A sign flip is two legs."""
    if before >= 0 and after >= 0:
        if after > before:
            return [("BUY", "open" if before == 0 else "add", before, after)]
        return [("SELL", "close" if after == 0 else "reduce", before, after)]
    if before <= 0 and after <= 0:
        if after < before:
            return [("SHORT", "open" if before == 0 else "add", before, after)]
        return [("COVER", "close" if after == 0 else "reduce", before, after)]
    if before > 0 > after:
        return [("SELL", "flip", before, 0.0), ("SHORT", "flip", 0.0, after)]
    return [("COVER", "flip", before, 0.0), ("BUY", "flip", 0.0, after)]


def leakage_check(
    close: pd.DataFrame, qv: pd.DataFrame, cfg: StrategyConfig, start: pd.Timestamp, full: np.ndarray
) -> tuple[bool, str]:
    """Recompute the targets with the later half of the session removed: earlier rows must not change."""
    idx = close.index
    s = idx.get_loc(start)
    cut = s + (len(idx) - s) // 2
    if cut <= s + 1:
        return True, "session too short to split; causality is covered by tests/test_book.py"
    trunc = sum(book_components(close.iloc[:cut], qv.iloc[:cut], cfg.book, cfg.book.allow_short).values())
    diff = float(np.max(np.abs(trunc[s:cut] - full[s:cut]))) if cut > s else 0.0
    ok = diff < 1e-9
    return ok, (
        f"targets recomputed without the {len(idx) - cut} hours after {idx[cut]:%Y-%m-%d %H:%M} UTC: "
        f"max difference {diff:.2e} ({'identical' if ok else 'CHANGED -> look-ahead'})"
    )


def run_session(
    close: pd.DataFrame,
    qv: pd.DataFrame,
    cfg: StrategyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    capital: float | None = None,
    check_leakage: bool = True,
) -> SessionResult:
    """Run the book from `start` (flat, in cash) to `end`. close/qv must already stop at `end`."""
    assert close.index[-1] <= end, "panel extends past the session end: refusing to run (leakage)"
    bt, ex, br = cfg.backtest, cfg.execution, cfg.breaker
    capital = capital or bt.capital
    comps = book_components(close, qv, cfg.book, cfg.book.allow_short)
    W = sum(comps.values())
    ok, detail = leakage_check(close, qv, cfg, start, W) if check_leakage else (True, "skipped")

    idx, pairs = close.index, list(close.columns)
    px = close.ffill(limit=3).to_numpy()
    s0 = idx.get_loc(start)
    hours = len(idx) - 1 - s0
    N = len(pairs)
    qty, prev = np.zeros(N), np.zeros(N)
    cash, peak, brake = capital, capital, 1.0
    cost = bt.fee + bt.slippage
    equity, expo, trades = [], [], []
    pnl = np.zeros(N)
    for s in range(hours + 1):
        t = s0 + s
        p = px[t]
        okp = np.isfinite(p) & (p > 0)
        value = np.where(okp, qty * np.nan_to_num(p), 0.0)
        e = cash + value.sum()
        equity.append(e)
        expo.append(
            {
                "gross": np.abs(value).sum() / e,
                "net": value.sum() / e,
                "long": value[value > 0].sum() / e,
                "short": -value[value < 0].sum() / e,
            }
        )
        if s == hours:
            break
        peak = max(peak, e)
        dd = 1.0 - e / peak
        if dd >= br.dd_level:
            brake = br.dd_mult
        elif dd <= br.dd_recover:
            brake = 1.0
        act = idx[t].hour in ex.decision_hours_utc
        for i in range(N):
            if not okp[i]:
                prev[i] = 0.0
                continue
            tgt = W[t, i] * brake
            cur = value[i] / e
            changed = abs(tgt - prev[i]) > 1e-9
            prev[i] = tgt
            if not ((changed or s == 0 or act) and abs(tgt - cur) > 1e-9):
                continue
            if (
                tgt != 0.0
                and cur != 0.0
                and abs(tgt - cur) < max(ex.min_trade_weight, ex.min_trade_frac * abs(tgt))
            ):
                continue  # resize too small to pay for itself
            notional_total = -qty[i] * p[i] if tgt == 0.0 else (tgt - cur) * e
            new_qty = 0.0 if tgt == 0.0 else qty[i] + notional_total / p[i]
            fee_total = abs(notional_total) * cost
            drivers = {k: float(comps[k][t, i] * brake) for k in SLEEVES if abs(comps[k][t, i]) > 1e-12}
            legs = _classify(cur, 0.0 if tgt == 0.0 else new_qty * p[i] / e)
            for action, reason, w0, w1 in legs:
                leg_notional = (w1 - w0) * e
                share = abs(leg_notional) / abs(notional_total) if notional_total else 0.0
                trades.append(
                    Trade(
                        idx[t] + HOUR,
                        pairs[i],
                        action,
                        reason,
                        abs(leg_notional) / p[i],
                        p[i],
                        abs(leg_notional),
                        fee_total * share,
                        w0,
                        w1,
                        drivers,
                    )
                )
            pnl[i] -= fee_total
            cash -= notional_total + fee_total
            qty[i] = new_qty
    # P&L per pair = cash flows + final holdings value (fees included above)
    flows = pd.Series(0.0, index=pairs)
    for tr in trades:
        sign = -1.0 if tr.action in ("BUY", "COVER") else 1.0
        flows[tr.pair] += sign * tr.notional
    final_px = pd.Series(px[s0 + hours], index=pairs).fillna(0.0)
    pnl_by_pair = flows + pd.Series(qty, index=pairs) * final_px + pd.Series(pnl, index=pairs)
    marks = idx[s0 : s0 + hours + 1] + HOUR
    return SessionResult(
        pd.Series(equity, index=marks),
        trades,
        pd.DataFrame(expo, index=marks),
        pnl_by_pair[pnl_by_pair.abs() > 0.005].sort_values(),
        ok,
        detail,
    )


def metrics(res: SessionResult, capital: float) -> dict:
    """Contest score and its parts on daily marks from the start (non-annualised), plus the usual
    return, drawdown, activity and cost figures."""
    eq = res.equity
    marks = eq.iloc[::24]
    if (len(eq) - 1) % 24:
        marks = pd.concat([marks, eq.iloc[-1:]])
    r = marks.pct_change().dropna()
    mean = r.mean() if len(r) else 0.0
    down = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    mdd_daily = float((1 - marks / marks.cummax()).max())
    sortino = mean / max(down, 1e-8)
    sharpe = mean / max(float(r.std(ddof=0)) if len(r) else 0.0, 1e-8)
    calmar = mean / max(mdd_daily, 1e-8)
    # contest days are 24h periods from the start, not calendar dates
    days_traded = len({int((t.time - eq.index[0]) // pd.Timedelta(days=1)) for t in res.trades})
    n_days = max(1, int(round((eq.index[-1] - eq.index[0]) / pd.Timedelta(days=1))))
    # previous edition's leaderboard: Calmar = total return / max drawdown, active days = dates with a trade
    total = float(eq.iloc[-1] / capital - 1)
    mdd_hourly = float((1 - eq / eq.cummax()).max())
    calmar_total = total / max(mdd_hourly, 1e-8)
    sharpe_lb = mean / max(float(r.std(ddof=1)) if len(r) > 1 else 0.0, 1e-8)
    active_dates = len({t.time.date() for t in res.trades})
    return {
        "start_equity": capital,
        "end_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / capital - 1),
        "score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
        "sortino": sortino,
        "sharpe": sharpe,
        "calmar": calmar,
        "sharpe_annualised": sharpe * np.sqrt(365),
        "sortino_annualised": sortino * np.sqrt(365),
        "max_drawdown_daily": mdd_daily,
        "max_drawdown_hourly": float((1 - eq / eq.cummax()).max()),
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
        "positive_days": f"{int((r > 0).sum())}/{len(r)}",
        "trades": len(res.trades),
        "active_trading_days": f"{days_traded}/{n_days}",
        "fees_paid": float(sum(t.fee for t in res.trades)),
        "turnover_x_capital": float(sum(t.notional for t in res.trades) / capital),
        "avg_gross_exposure": float(res.exposure["gross"].mean()),
        "avg_net_exposure": float(res.exposure["net"].mean()),
        "max_gross_exposure": float(res.exposure["gross"].max()),
        "daily_returns": r,
        "calmar_total": calmar_total,
        "active_dates": active_dates,
        "rank_vs_field": rank_vs_field(sharpe_lb, sortino, calmar_total, active_dates),
        "winner": WINNER,
    }
