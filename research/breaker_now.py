"""Does the live drawdown breaker cost the bear-mode rebound its upside, starting from today's drawdown?

Rounds 19-21 ran the simulator with the breaker off and every window starting at its own peak. Live, the
account is already ~7% under its peak, so a further ~3% loss halves the book (dd_level 0.10, dd_mult 0.5)
until it recovers to 5% under the peak. Same 699 post-crash starts and objective; the simulator below is
backtest/contest.py's `_simulate` with one change: the window starts `dd0` under its peak.

    python research/breaker_now.py [hours_left]
"""

import sys
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

H = int(sys.argv[1]) if len(sys.argv) > 1 else 220


@nb.njit(cache=True)
def sim(W, px, act, starts, H, cost, min_w, min_frac, dd0, dd_lvl, dd_mult, dd_rec):
    """contest._simulate with capital 1 and a starting peak of 1 / (1 - dd0); returns final equity."""
    N = W.shape[1]
    final = np.empty(starts.shape[0])
    for k in range(starts.shape[0]):
        st = starts[k]
        qty = np.zeros(N)
        prev = np.zeros(N)
        cash = 1.0
        peak = 1.0 / (1.0 - dd0)  # the live account's peak, above today's equity
        brake = 1.0
        for s in range(H + 1):
            t = st + s
            e = cash
            for i in range(N):
                p = px[t, i]
                if p > 0.0 and np.isfinite(p):
                    e += qty[i] * p
            if s == H:
                final[k] = e
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
    return final


def main() -> None:
    lab = Lab()
    b, bm, br, ex = lab.cfg.book, lab.cfg.book.bear_mode, lab.cfg.breaker, lab.cfg.execution
    off = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
    normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
    bear = B.bear_regime(lab.close, lab.qv, bm)[:, None]
    books = {
        "old bot": normal,
        "rebound switch": np.where(bear, np.nan_to_num(B.rebound_sleeve(lab.close, lab.qv, bm)), normal),
    }
    px = lab.close.ffill(limit=3)
    liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
    ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
    act = np.isin(lab.idx.hour, ex.decision_hours_utc)
    first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
    starts = np.arange(first, len(lab.idx) - H - 1, 6)
    prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
    idx_m = starts[prior < bm.enter]
    ep = np.cumsum(np.r_[1, np.diff(idx_m) > 72])
    levels = {"live 0.10": br.dd_level, "0.15": 0.15, "0.20": 0.20, "off": 9.0}
    rows = []
    for dd0 in (0.05, 0.07, 0.09):
        for bn, W in books.items():
            Wc = np.ascontiguousarray(W)
            for ln, lvl in levels.items():
                r = (
                    sim(
                        Wc,
                        lab.px,
                        act,
                        idx_m,
                        H,
                        lab.cost,
                        ex.min_trade_weight,
                        ex.min_trade_frac,
                        dd0,
                        lvl,
                        br.dd_mult,
                        br.dd_recover,
                    )
                    - 1.0
                )
                rows.append(
                    {
                        "now under peak": f"{dd0:.0%}",
                        "book": bn,
                        "breaker": ln,
                        "P>4.5%": np.mean(r > 0.045),
                        "P>6%": np.mean(r > 0.06),
                        "episode P>4.5%": np.mean([np.mean(r[ep == e] > 0.045) for e in np.unique(ep)]),
                        "median": np.median(r),
                        "P<-8%": np.mean(r < -0.08),
                        "P<-15%": np.mean(r < -0.15),
                    }
                )
    print(
        f"{len(idx_m)} post-crash starts in {ep.max()} episodes, {H} hours left; returns from today's equity"
    )
    with pd.option_context(
        "display.width", 250, "display.max_rows", 100, "display.float_format", "{:.3f}".format
    ):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
