"""Every bStock idea side by side, scored fortnight by fortnight.

Fortnights: back-to-back 14-day windows starting 12:00 UTC every 14 days from 15 Feb 2024 (69 of them),
each from $100k cash, 0.15% per side, the live trade rule and breaker (backtest/contest.py).
Score: the contest composite on daily marks (non-annualised). Three views:
  A. all fortnights, stocks priced with their US underlying (research/fetch_stock_proxies.py)
  B. 2026 fortnights only, same prices
  C. after listing (1 Jul - 15 Sep 2026) at the real 24/7 token prices; only 5 back-to-back
     fortnights exist, so the daily-start windows (77) are shown too. This is the only view free of
     the selection bias in the stock list (research/stocks_sleeve_validation.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import stocks_hedge_experiment as hx  # noqa: E402  (hedge variants, panels)
import stocks_sleeve_experiment as sx  # noqa: E402  (stock sleeves, live book)
from backtest.contest import contest_score, simulate  # noqa: E402
from trend_guard.strategy.book import momentum_sleeve, trend_sleeve  # noqa: E402
from trend_guard.strategy.team_book import team_momentum, team_trend  # noqa: E402

close, idx, ncr, b = sx.close, sx.close.index, sx.ncr, sx.cfg.book
qv = pd.concat([sx.c_qv, sx.s_qv], axis=1)


def book_with_stocks(stock_sleeves: set[str]) -> np.ndarray:
    """The live book, but the named sleeves see crypto + stocks (others crypto only)."""
    shares = {"v2_trend": b.v2_share * b.trend_share, "v2_momentum": b.v2_share * b.momentum_share,
              "team_trend": b.team_share * b.team.trend_share, "team_momentum": b.team_share * b.team.momentum_share}
    total = np.zeros(close.shape)
    for k, s in shares.items():
        cols = list(close.columns) if k in stock_sleeves else list(sx.c_close.columns)
        C, Q = close[cols], qv[cols]
        W = {"v2_trend": lambda: trend_sleeve(C, Q, b.trend), "v2_momentum": lambda: momentum_sleeve(C, Q, b.momentum, b.allow_short),
             "team_trend": lambda: team_trend(C, Q), "team_momentum": lambda: team_momentum(C, Q, b.team.momentum_vol_target, b.allow_short)}[k]()
        total[:, [close.columns.get_loc(c) for c in cols]] += s * W
    total *= b.risk_multiplier
    g = np.abs(total).sum(1, keepdims=True)
    return np.nan_to_num(total * np.where(g > b.max_gross, b.max_gross / np.maximum(g, 1e-12), 1.0))


def hold(pairs: list[str], weights: list[float]) -> np.ndarray:
    W = np.zeros(close.shape)
    for p, w in zip(pairs, weights):
        W[:, close.columns.get_loc(p)] = np.where(close[p].notna(), w, 0.0)
    return W


stocks_all = list(sx.s_close.columns)
S_mom, S_tr = sx.stock_momentum(0.015), sx.stock_trend(0.004)
variants = {
    "Crypto only (live book, main)": hx.variants["live book"],
    "Book, stocks allowed in trend sleeves": book_with_stocks({"v2_trend", "team_trend"}),
    "Book, stocks allowed in all sleeves": book_with_stocks({"v2_trend", "v2_momentum", "team_trend", "team_momentum"}),
    "80% book + 20% stock momentum sleeve": sx.assemble(S_mom, 0.2, "carve"),
    "80% book + 20% stock trend sleeve": sx.assemble(S_tr, 0.2, "carve"),
    "Book + stock momentum on idle capital": sx.assemble(S_mom, 1.0, "add"),
    "Book + COIN/MSTR short hedge (25%)": hx.variants["beta hedge 0.25 via COINB/MSTRB"],
    "Book + stock hedge when BTC weak (50%)": hx.variants["beta hedge 0.5 stocks, risk-off only"],
    "Book + idle cash in MSFT/GOOGL/META": hx.variants["defensive park 0.5 (MSFT/GOOGL/META)"],
    "Hold BTC": hold(["BTC/USD"], [0.999]),
}


def fortnight_starts(first: str, last: str, step_days: int) -> np.ndarray:
    out = []
    for s in pd.date_range(f"{first} 12:00", f"{last} 12:00", freq=f"{step_days}D", tz="UTC"):
        if s in idx and idx.get_loc(s) + 14 * 24 < len(idx):
            out.append(idx.get_loc(s))
    return np.array(out, dtype=np.int64)


def table(title, prices, starts, rows_mask=None):
    print(f"\n{title}")
    print(f"{'':42s}{'score':>7s}{'median':>8s}{'>-1%':>6s}{'pos':>6s}{'>+5%':>6s}{'worst':>8s}{'best':>8s}")
    P = prices.to_numpy()
    for name, W in variants.items():
        W = np.where(np.isfinite(P), W, 0.0)
        eq, _ = simulate(W, prices, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
        r, s = eq[:, -1] / eq[:, 0] - 1, contest_score(eq)
        print(f"{name:42s}{np.median(s):7.3f}{np.median(r):+8.2%}{(r > -0.01).mean():6.0%}{(r > 0).mean():6.0%}{(r > 0.05).mean():6.0%}{r.min():+8.1%}{r.max():+8.1%}")


if __name__ == "__main__":
    fn = fortnight_starts("2024-02-15", "2026-09-15", 14)
    table(f"A. All back-to-back fortnights, Feb 2024 - Sep 2026 ({len(fn)}), stocks at underlying prices", close, fn)
    fn26 = fn[idx[fn] >= pd.Timestamp("2026-01-01", tz="UTC")]
    table(f"B. 2026 back-to-back fortnights ({len(fn26)}), stocks at underlying prices", close, fn26)
    tok = {}
    for f in sorted((ROOT / "data/bars_stocks/1h").glob("*.parquet")):
        d = pd.read_parquet(f)["close"]; d.index = pd.to_datetime(d.index, utc=True)
        tok[f.stem.replace("USDT", "") + "/USD"] = d
    close_tok = pd.concat([sx.c_close, pd.DataFrame(tok).reindex(idx)[stocks_all]], axis=1)
    fnt = fortnight_starts("2026-07-01", "2026-09-15", 14)
    table(f"C1. After listing, back-to-back fortnights ({len(fnt)}), real token prices", close_tok, fnt)
    daily = fortnight_starts("2026-07-01", "2026-09-15", 1)
    table(f"C2. After listing, fortnights starting every day ({len(daily)}, overlapping), real token prices", close_tok, daily)
