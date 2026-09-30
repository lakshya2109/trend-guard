"""Can tokenized stocks cushion a sharp crypto crash without hurting the book in normal times?

Part 1, event study: every sharp BTC drop (3-day return below -10%, episodes at least 7 days apart),
and what the stocks did over the same 72 hours (US underlying prices).
Part 2, crash-only cushions: switched on only while BTC's 72-hour return is below -6% (checked at the
00:00 / 12:00 UTC decision hours, held until the next check), so they cost nothing when quiet:
  short COIN/MSTR (50% of net long crypto) | short BTC instead | halve crypto to cash |
  halve crypto into MSFT/GOOGL/META | plus a permanent 15% MSFT/GOOGL/META holding for comparison.
Scored on the 944 daily-start windows, split into crash windows (BTC falls >= 15% peak-to-trough inside
the window) and normal windows; then at real token prices after listing (1 Jul 2026 on).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import stocks_sleeve_experiment as sx  # noqa: E402
from backtest.contest import contest_score, simulate, window_starts  # noqa: E402

close, idx, ncr, book = sx.close, sx.close.index, sx.ncr, sx.book
btc = sx.c_close["BTC/USD"]
DEF = ["MSFTB/USD", "GOOGLB/USD", "METAB/USD"]
BETA = ["COINB/USD", "MSTRB/USD"]
basket_all = list(sx.s_close.columns)

# ------------------------------------------------------------------ part 1: event study
r72 = btc / btc.shift(72) - 1
events, last = [], None
for t in r72[r72 < -0.10].index:
    if last is None or t - last > pd.Timedelta(days=7):
        # the 72h window ending at the worst point of this episode
        seg = r72.loc[t: t + pd.Timedelta(days=3)]
        end = seg.idxmin(); events.append(end); last = end
print("Part 1. Sharp BTC drops (3-day return below -10%) and the stocks over the same 72 hours")
print(f"{'window end (UTC)':22s}{'BTC':>7s}{'crypto book':>12s}{'all stocks':>11s}{'MSFT/GOOGL/META':>16s}{'COIN/MSTR':>10s}")
bk_eq = pd.Series(1 + np.nansum(book[:-1] * sx.c_close.pct_change(fill_method=None).shift(-1).to_numpy()[:-1], axis=1),
                  index=idx[:-1]).cumprod()
def ret(cols, a, b):
    p = close.loc[[a, b], cols]; r = (p.iloc[1] / p.iloc[0] - 1).dropna(); return r.mean() if len(r) else np.nan
rows = []
for end in events:
    a = end - pd.Timedelta(hours=72)
    if a < close.index[0] or end not in close.index:
        continue
    row = (btc[end] / btc[a] - 1, bk_eq.get(end, np.nan) / bk_eq.get(a, np.nan) - 1, ret(basket_all, a, end), ret(DEF, a, end), ret(BETA, a, end))
    rows.append(row)
    print(f"{end:%Y-%m-%d %H:%M}      {row[0]:+7.1%}{row[1]:+12.1%}{row[2]:+11.1%}{row[3]:+16.1%}{row[4]:+10.1%}")
R = np.array(rows)
print(f"{'average':22s}{R[:,0].mean():+7.1%}{np.nanmean(R[:,1]):+12.1%}{np.nanmean(R[:,2]):+11.1%}{np.nanmean(R[:,3]):+16.1%}{np.nanmean(R[:,4]):+10.1%}")

# ------------------------------------------------------------------ part 2: crash-only cushions
decide = pd.Series(np.isin(idx.hour, [0, 12]), index=idx)
trigger = pd.Series(np.where(decide, (r72 < -0.06).reindex(idx).fillna(False).astype(float), np.nan), index=idx).ffill().fillna(0).to_numpy()
net_long = np.clip(book.sum(1), 0, None)


def build(kind, share=0.5):
    W = np.zeros(close.shape); W[:, :ncr] = book
    on = trigger[:, None] > 0
    if kind == "short_stocks":
        for p in BETA:
            W[:, close.columns.get_loc(p)] -= trigger * net_long * share / len(BETA) * close[p].notna().to_numpy()
    elif kind == "short_btc":
        W[:, close.columns.get_loc("BTC/USD")] -= trigger * net_long * share
    elif kind in ("to_cash", "to_defensive"):
        cut = np.where(on, W[:, :ncr] * share, 0.0)
        W[:, :ncr] -= cut
        if kind == "to_defensive":
            freed = np.abs(cut).sum(1)
            for p in DEF:
                W[:, close.columns.get_loc(p)] += freed / len(DEF)
    elif kind == "always_defensive":
        W[:, :ncr] *= 1 - share
        for p in DEF:
            W[:, close.columns.get_loc(p)] += share / len(DEF)
    g = np.abs(W).sum(1, keepdims=True)
    return np.nan_to_num(W * np.where(g > 1.0, 1.0 / np.maximum(g, 1e-12), 1.0))


variants = {"crypto only (live book)": build("none"),
            "crash: short COIN/MSTR (50%)": build("short_stocks"),
            "crash: short BTC (50%)": build("short_btc"),
            "crash: halve crypto -> cash": build("to_cash"),
            "crash: halve crypto -> MSFT/GOOGL/META": build("to_defensive"),
            "always: 15% in MSFT/GOOGL/META": build("always_defensive", 0.15)}


def run(prices, starts):
    P = prices.to_numpy(); out = {}
    for name, W in variants.items():
        eq, _ = simulate(np.where(np.isfinite(P), W, 0.0), prices, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
        out[name] = (eq[:, -1] / eq[:, 0] - 1, contest_score(eq))
    return out


if __name__ == "__main__":
    print(f"\ncrash trigger (BTC 72h < -6%) is on {trigger[idx >= pd.Timestamp('2024-02-15', tz='UTC')].mean():.1%} of hours")
    starts = sx.starts
    btc_dd = np.array([(btc.iloc[s:s + 337] / btc.iloc[s:s + 337].cummax() - 1).min() for s in starts])
    crash = btc_dd <= -0.15
    res = run(close, starts)
    print(f"\nPart 2. 944 daily-start windows, underlying prices: {crash.sum()} crash windows (BTC falls >= 15% inside), {(~crash).sum()} normal")
    print(f"{'':40s}{'CRASH: median':>14s}{'mean':>7s}{'worst':>7s}{'score':>7s}   {'NORMAL: median':>15s}{'score':>7s}   {'ALL score':>9s}")
    for name, (r, s) in res.items():
        print(f"{name:40s}{np.median(r[crash]):+14.2%}{r[crash].mean():+7.2%}{r[crash].min():+7.1%}{np.median(s[crash]):7.3f}   "
              f"{np.median(r[~crash]):+15.2%}{np.median(s[~crash]):7.3f}   {np.median(s):9.3f}")
    tok = {}
    for f in sorted((ROOT / "data/bars_stocks/1h").glob("*.parquet")):
        d = pd.read_parquet(f)["close"]; d.index = pd.to_datetime(d.index, utc=True)
        tok[f.stem.replace("USDT", "") + "/USD"] = d
    close_tok = pd.concat([sx.c_close, pd.DataFrame(tok).reindex(idx)[basket_all]], axis=1)
    st2 = window_starts(close_tok.index, sx.cfg.backtest.model_copy(update={"first_window": "2026-07-01"}))
    dd2 = np.array([(btc.iloc[s:s + 337] / btc.iloc[s:s + 337].cummax() - 1).min() for s in st2])
    res2 = run(close_tok, st2)
    print(f"\nAfter listing, real token prices: {len(st2)} windows, worst BTC fall inside a window {dd2.min():.1%} "
          f"(crash windows: {(dd2 <= -0.15).sum()}), trigger on {trigger[idx >= pd.Timestamp('2026-07-01', tz='UTC')].mean():.1%} of hours")
    print(f"{'':40s}{'median':>8s}{'worst':>7s}{'score':>7s}")
    for name, (r, s) in res2.items():
        print(f"{name:40s}{np.median(r):+8.2%}{r.min():+7.1%}{np.median(s):7.3f}")
