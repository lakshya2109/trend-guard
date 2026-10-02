"""Execution-aware contest simulator: patient maker limits for spot trades, market for shorts.

Same trade rule as backtest/contest.py. The difference is HOW a spot trade fills:
  - decided at the close of bar t, a limit is posted at close_t x (1 -/+ offset) (buy / sell);
  - it fills at its limit price, maker fee, if a later bar trades THROUGH it by `thru` (low <= L(1-thru)
    for a buy), a conservative stand-in for the mock exchange's "fills when the market reaches your price";
  - each hour it is not filled it is re-posted at the new close (the live bot reprices), and after
    `patience` hours the residual goes at market: close x (1 + slip), taker fee;
  - patience = 0 means every trade is a market order at the decision close (the old model).
Shorts (/v6) pay the flat short fee on open and close and fill at market, as on Roostoo.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd


@nb.njit(cache=True)
def _sim(
    W,
    px,
    hi,
    lo,
    act,
    starts,
    H,
    maker,
    taker,
    short_fee,
    slip,
    offset,
    thru,
    patience,
    min_w,
    min_frac,
    capital,
):
    nwin = starts.shape[0]
    N = W.shape[1]
    equity = np.empty((nwin, H + 1))
    turnover = np.zeros(nwin)
    fees = np.zeros(nwin)
    maker_share = np.zeros(nwin)
    for k in range(nwin):
        st = starts[k]
        qty = np.zeros(N)
        prev = np.zeros(N)
        pend = np.zeros(N)  # pending spot order, in coin units (+ buy, - sell)
        lim = np.zeros(N)
        age = np.zeros(N, dtype=np.int64)
        cash = capital
        made = 0.0
        tot = 0.0
        for s in range(H + 1):
            t = st + s
            # 1) resting limits from the previous hour: fill if this bar traded through them
            for i in range(N):
                if pend[i] == 0.0:
                    continue
                p = px[t, i]
                if not (p > 0.0 and np.isfinite(p)):
                    pend[i] = 0.0
                    continue
                filled = False
                if pend[i] > 0 and lo[t, i] <= lim[i] * (1.0 - thru):
                    filled = True
                elif pend[i] < 0 and hi[t, i] >= lim[i] * (1.0 + thru):
                    filled = True
                if filled:
                    n = pend[i] * lim[i]
                    cash -= n + abs(n) * maker
                    qty[i] += pend[i]
                    fees[k] += abs(n) * maker
                    made += abs(n)
                    tot += abs(n)
                    turnover[k] += abs(n) / capital
                    pend[i] = 0.0
                else:
                    age[i] += 1
                    if age[i] >= patience:  # out of patience: cross the spread now
                        n = pend[i] * p * (1.0 + slip * np.sign(pend[i]))
                        cash -= n + abs(n) * taker
                        qty[i] += pend[i]
                        fees[k] += abs(n) * taker
                        tot += abs(n)
                        turnover[k] += abs(n) / capital
                        pend[i] = 0.0
                    else:
                        lim[i] = p * (1.0 - offset * np.sign(pend[i]))  # reprice at the new close
            e = cash
            for i in range(N):
                p = px[t, i]
                if p > 0.0 and np.isfinite(p):
                    e += qty[i] * p
            equity[k, s] = e
            if s == H:
                break
            # 2) the contest trade rule, against holdings (pending orders are replaced, not stacked)
            for i in range(N):
                p = px[t, i]
                ok = p > 0.0 and np.isfinite(p)
                tgt = W[t, i] if ok else 0.0
                if not np.isfinite(tgt):
                    tgt = 0.0
                cur = qty[i] * p / e if ok else 0.0
                changed = abs(tgt - prev[i]) > 1e-9
                prev[i] = tgt
                if not ((changed or s == 0 or act[t]) and ok and abs(tgt - cur) > 1e-9):
                    continue
                delta = tgt - cur
                if tgt != 0.0 and cur != 0.0 and abs(delta) < max(min_w, min_frac * abs(tgt)):
                    pend[i] = 0.0  # already close enough: cancel any resting order
                    continue
                dq = -qty[i] if tgt == 0.0 else delta * e / p
                spot = (cur >= 0.0 and tgt >= 0.0) and qty[i] >= 0.0
                if spot and patience > 0:
                    if pend[i] == 0.0 or np.sign(pend[i]) != np.sign(dq):
                        age[i] = 0
                    pend[i] = dq
                    lim[i] = p * (1.0 - offset * np.sign(dq))
                    continue
                # market now: shorts, flips, or the old model
                pend[i] = 0.0
                n = dq * p * (1.0 + slip * np.sign(dq))
                fee = short_fee if not spot else taker
                if patience == 0 and spot:
                    fee = taker
                cash -= n + abs(n) * fee
                qty[i] += dq
                fees[k] += abs(n) * fee
                tot += abs(n)
                turnover[k] += abs(n) / capital
        maker_share[k] = made / tot if tot > 0 else 0.0
    return equity, turnover, fees, maker_share


def load_hilo(cols: list[str], index: pd.DatetimeIndex, bars_dir) -> tuple[np.ndarray, np.ndarray]:
    hi = pd.DataFrame(np.nan, index=index, columns=cols)
    lo = hi.copy()
    for c in cols:
        f = bars_dir / f"{c.split('/')[0]}USDT.parquet"
        d = pd.read_parquet(f)
        hi[c] = d["high"].reindex(index)
        lo[c] = d["low"].reindex(index)
    return hi.ffill(limit=3).to_numpy(), lo.ffill(limit=3).to_numpy()


def simulate_exec(
    lab,
    W: np.ndarray,
    starts: np.ndarray,
    H: int,
    *,
    patience: int = 0,
    offset: float = 0.0,
    thru: float = 0.001,
    maker: float = 0.0005,
    taker: float = 0.001,
    short_fee: float = 0.001,
    slip: float = 0.0005,
    capital: float = 1.0,
):
    if not hasattr(lab, "hi"):
        from research.harness import ROOT

        lab.hi, lab.lo = load_hilo(lab.cols, lab.idx, ROOT / "data/bars/1h")
    ex = lab.cfg.execution
    act = np.isin(lab.idx.hour, ex.decision_hours_utc)
    return _sim(
        np.ascontiguousarray(np.nan_to_num(W), dtype=np.float64),
        lab.px,
        lab.hi,
        lab.lo,
        act,
        starts.astype(np.int64),
        H,
        maker,
        taker,
        short_fee,
        slip,
        offset,
        thru,
        patience,
        ex.min_trade_weight,
        ex.min_trade_frac,
        capital,
    )
