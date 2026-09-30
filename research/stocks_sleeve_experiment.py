"""A dedicated tokenized-stock sleeve next to the live crypto book.

Stocks are priced with their US underlying (Yahoo hourly, research/fetch_stock_proxies.py), forward-
filled across nights and weekends, which gives history back to Nov 2023 (15 of 21 names; the rest
join as they list). research/stocks_proxy_check.py: the proxy's 14-day returns correlate 0.95 with the
real 24/7 tokens, but 38% of token variance happens outside US hours, so these paths are smoother
than live. Every run: the contest replay on all 944 windows (12:00 UTC, $100k, 0.15% per side).

Sleeves (long-only; shorting stocks into the US open is what hurt in research/stocks_experiment.py):
  trend     v2's trend sleeve machinery on the stock panel (t-stat trend, low vol target, daily)
  momentum  v2's momentum sleeve machinery on stocks, long leg only
Sizing: 'carve' = (1 - s) x live book + s x sleeve; 'add' = live book + s x sleeve, gross capped at 1.
Adoption rule (round 8): median score up in 2024-25 and 2026, above -1% share down <= 2 pts, worst no worse.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.contest import simulate, summarize, window_starts  # noqa: E402
from backtest.run_backtest import PERIODS, load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import book_weights, momentum_sleeve, trend_sleeve  # noqa: E402

cfg = load_strategy_config(ROOT / "config/strategy.yaml"); sym = load_symbols(ROOT / "config/symbols.yaml")


def stock_panel(index):
    close, qv = {}, {}
    for f in sorted((ROOT / "data/bars_proxy/1h").glob("*.parquet")):
        d = pd.read_parquet(f); d.index = pd.to_datetime(d.index, utc=True)
        pair = f.stem.replace("USDT", "") + "/USD"
        first = d.index[0]
        close[pair] = d["close"].reindex(d.index.union(index)).ffill().reindex(index).where(index >= first)
        # daily dollar volume spread over the day so the 7-day liquidity rule sees a sensible number
        qv[pair] = d["quote_volume"].reindex(index).fillna(0.0).where(index >= first)
    return pd.DataFrame(close), pd.DataFrame(qv)


c_close, c_qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
s_close, s_qv = stock_panel(c_close.index)
close = pd.concat([c_close, s_close], axis=1); ncr = c_close.shape[1]
book = book_weights(c_close, c_qv, cfg.book, cfg.book.allow_short).to_numpy()
starts = window_starts(close.index, cfg.backtest); st = close.index[starts]


def stock_trend(target):
    p = cfg.book.trend.model_copy(update={"target_vol_daily": target, "min_age_h": 24 * 30, "exclude": []})
    return trend_sleeve(s_close, s_qv, p)


def stock_momentum(target, k=4):
    p = cfg.book.momentum.model_copy(update={"target_vol_daily": target, "k": k, "buffer": 2 * k, "exclude": [],
                                             "min_age_h": 24 * 30, "short_crash_return": -1.0})
    return momentum_sleeve(s_close, s_qv, p, allow_short=False)


def assemble(sleeve, s, mode):
    W = np.zeros(close.shape)
    if mode == "carve":
        W[:, :ncr] = (1 - s) * book; W[:, ncr:] = s * sleeve
    else:
        W[:, :ncr] = book; W[:, ncr:] = s * sleeve
        g = np.abs(W).sum(1, keepdims=True); W *= np.where(g > cfg.book.max_gross, cfg.book.max_gross / np.maximum(g, 1e-12), 1.0)
    return np.nan_to_num(W)


def evaluate(W):
    eq, turn = simulate(W, close, starts, cfg.backtest, cfg.execution, cfg.breaker)
    t = summarize(eq, turn, st, PERIODS)
    return {p: (t.loc[p, "score"], t.loc[p, "median_return"], t.loc[p, "above_-1%"], t.loc[p, "worst"]) for p in ("2024-25", "2026", "all")}


if __name__ == "__main__":
    W0 = assemble(np.zeros(s_close.shape), 0.0, "add")
    base = evaluate(W0)
    print(f"book gross (mean, 2024-26): {np.abs(book[starts[0]:]).sum(1).mean():.2f}x equity -> idle capital for an 'add' sleeve")
    fmt = lambda r, p: f"{r[p][0]:.3f} {r[p][1]:+.2%} {r[p][2]:.0%} {r[p][3]:+.1%}"
    print(f"{'':40s}{'2024-25: score med >-1% worst':>31s}{'2026':>31s}{'all':>31s}  rule")
    print(f"{'live book (crypto only)':40s}{fmt(base,'2024-25'):>31s}{fmt(base,'2026'):>31s}{fmt(base,'all'):>31s}")
    sleeves = {"trend 0.4%/d": stock_trend(0.004), "trend 1.0%/d": stock_trend(0.010),
               "momentum top4 1.0%/d": stock_momentum(0.010), "momentum top4 1.5%/d": stock_momentum(0.015)}
    for sname, S in sleeves.items():
        for mode, shares in (("add", (0.25, 0.5, 1.0)), ("carve", (0.1, 0.2))):
            for s in shares:
                r = evaluate(assemble(S, s, mode))
                ok = (r["2024-25"][0] > base["2024-25"][0] and r["2026"][0] > base["2026"][0]
                      and all(r[p][2] >= base[p][2] - 0.02 for p in ("2024-25", "2026")) and r["all"][3] >= base["all"][3] - 1e-9)
                print(f"{sname + f' {mode} {s:g}':40s}{fmt(r,'2024-25'):>31s}{fmt(r,'2026'):>31s}{fmt(r,'all'):>31s}  {'PASS' if ok else 'fail'}", flush=True)
