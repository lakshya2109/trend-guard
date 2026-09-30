"""Do wide Roostoo spreads eat the edge on some pairs? The backtest charges a flat 5 bp slippage, but
several pairs quote much wider on Roostoo (tick size on sub-cent coins). Samples live spreads, then
re-runs the full contest replay with pairs above a spread threshold removed from the universe.
"""
import sys, time
from pathlib import Path

import numba as nb

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.contest import contest_score, simulate, summarize, window_starts  # noqa: E402
from backtest.run_backtest import PERIODS, load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import book_weights  # noqa: E402

@nb.njit(cache=True)
def simulate_pair_costs(W, px, act, starts, H, cost, min_w, min_frac, capital, dd_lvl, dd_mult, dd_rec):
    """backtest/contest.py::_simulate with `cost` per pair (fee + that pair's slippage) instead of one scalar."""
    nwin = starts.shape[0]; N = W.shape[1]
    equity = np.empty((nwin, H + 1)); turnover = np.zeros(nwin)
    for k in range(nwin):
        st = starts[k]; qty = np.zeros(N); prev = np.zeros(N); cash = capital; peak = capital; brake = 1.0
        for s in range(H + 1):
            t = st + s; e = cash
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
                p = px[t, i]; ok = p > 0.0 and np.isfinite(p)
                tgt = W[t, i] * brake if ok else 0.0
                if not np.isfinite(tgt):
                    tgt = 0.0
                cur = qty[i] * p / e if ok else 0.0
                changed = abs(tgt - prev[i]) > 1e-9; prev[i] = tgt
                if not ((changed or s == 0 or act[t]) and ok and abs(tgt - cur) > 1e-9):
                    continue
                delta = tgt - cur
                if tgt != 0.0 and cur != 0.0 and abs(delta) < max(min_w, min_frac * abs(tgt)):
                    continue
                if tgt == 0.0:
                    notional = -qty[i] * p; qty[i] = 0.0
                else:
                    notional = delta * e; qty[i] += notional / p
                cash -= notional + abs(notional) * cost[i]
                turnover[k] += abs(notional) / capital
    return equity, turnover


def run(W, close, costs, cfg, starts):
    px = close.ffill(limit=3).to_numpy(dtype=np.float64); act = np.isin(close.index.hour, cfg.execution.decision_hours_utc)
    b, ex, br = cfg.backtest, cfg.execution, cfg.breaker
    return simulate_pair_costs(np.ascontiguousarray(W, dtype=np.float64), px, act, starts, b.window_days * 24,
                               np.ascontiguousarray(costs, dtype=np.float64), ex.min_trade_weight, ex.min_trade_frac,
                               b.capital, br.dd_level, br.dd_mult, br.dd_recover)


samples = []
for i in range(5):  # a few snapshots, 20 s apart, to avoid one noisy quote
    d = requests.get("https://mock-api.roostoo.com/v3/ticker", params={"timestamp": int(time.time() * 1000)}, timeout=20).json()["Data"]
    samples.append({p: (v["MinAsk"] - v["MaxBid"]) / ((v["MinAsk"] + v["MaxBid"]) / 2) * 1e4 for p, v in d.items() if v.get("MaxBid")})
    time.sleep(20)
spread = pd.DataFrame(samples).median()
cfg = load_strategy_config(ROOT / "config/strategy.yaml"); sym = load_symbols(ROOT / "config/symbols.yaml")
close, qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
sp = spread.reindex(close.columns)
print("widest Roostoo spreads (median of 5 snapshots, bp):")
print(sp.sort_values(ascending=False).head(15).round(1).to_string())
starts = window_starts(close.index, cfg.backtest); st = close.index[starts]
# cost per side = fee + max(configured slippage, half the pair's live Roostoo spread)
half = (sp.fillna(0) / 2 / 1e4).clip(lower=cfg.backtest.slippage)
cost_all = (cfg.backtest.fee + half).to_numpy()
print(f"\nper-pair cost per side: median {np.median(cost_all)*1e4:.1f} bp, max {cost_all.max()*1e4:.1f} bp (flat model: {(cfg.backtest.fee+cfg.backtest.slippage)*1e4:.1f} bp)")
for thr in (None, 20, 12, 8):
    keep = list(close.columns) if thr is None else [c for c in close.columns if not (sp.get(c, 0) > thr)]
    W = book_weights(close[keep], qv[keep], cfg.book, cfg.book.allow_short).to_numpy()
    label = "all pairs (current)" if thr is None else f"drop spread > {thr} bp ({len(close.columns) - len(keep)} pairs)"
    for model, eqt in (("flat 15 bp", simulate(W, close[keep], starts, cfg.backtest, cfg.execution, cfg.breaker)),
                       ("per-pair spread cost", run(W, close[keep], (cfg.backtest.fee + half.reindex(keep)).to_numpy(), cfg, starts))):
        t = summarize(eqt[0], eqt[1], st, PERIODS)
        print(f"{label:34s} {model:22s} " + "  ".join(f"{per}: score {t.loc[per,'score']:.3f} med {t.loc[per,'median_return']:+.2%} >-1% {t.loc[per,'above_-1%']:.0%} worst {t.loc[per,'worst']:+.1%}" for per in ("2024-25", "2026", "all")))
