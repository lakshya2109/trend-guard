"""Improving the crash cushion without fooling ourselves: a pre-declared grid, chosen on 2024-25 only,
confirmed on 2026 (unseen; it holds three sharp BTC drops), then crash-by-crash.

Grid (32 variants), all shorting crypto-linked stock tokens against net long crypto:
  trigger   A: BTC 72h return < -6% (current) | B: BTC 24h return < -4% | C: A or B
            | D: A and BTC 24h realized vol above its 30-day median (a crash, not a slow bleed)
  checked   at the 00/12 UTC decision hours (current) | every hour
  hold      off as soon as the trigger clears (current) | stay on at least 24h after the last trigger
  basket    COIN + MSTR (current) | COIN + MSTR + CRCL (Circle, from Jun 2025)
  size      75% of net long crypto (the best size in bstocks_crash_robustness.py)
Selection rule, fixed before running: among variants whose 2024-25 all-window median score is within
0.003 of crypto-only AND whose 2024-25 normal-window median return is within 0.05 pts, pick the one
with the best 2024-25 crash-window mean return. Then report that pick (and the current version) on 2026.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import bstocks_crash_cushion as cx  # noqa: E402
import stocks_sleeve_experiment as sx  # noqa: E402
from backtest.contest import contest_score, simulate  # noqa: E402

idx, close, btc = cx.idx, cx.close, cx.btc
r72 = (btc / btc.shift(72) - 1).reindex(idx)
r24 = (btc / btc.shift(24) - 1).reindex(idx)
lr = np.log(btc).diff()
rv24 = lr.rolling(24).std()
hot = (rv24 > rv24.rolling(30 * 24, min_periods=240).median()).reindex(idx).fillna(False)
RAW = {"A": (r72 < -0.06), "B": (r24 < -0.04), "C": (r72 < -0.06) | (r24 < -0.04), "D": (r72 < -0.06) & hot}
RAW = {k: v.reindex(idx).fillna(False).to_numpy() for k, v in RAW.items()}
decide = np.isin(idx.hour, [0, 12])


def trigger(kind, hourly, hold24):
    raw = RAW[kind].astype(float)
    if not hourly:
        raw = pd.Series(np.where(decide, raw, np.nan), index=idx).ffill().fillna(0).to_numpy()
    if hold24:
        raw = pd.Series(raw, index=idx).rolling(24, min_periods=1).max().to_numpy()
    return raw


def hedged(trig, basket, size=0.75):
    W = np.zeros(close.shape); W[:, :cx.ncr] = cx.book
    live = np.stack([close[p].notna().to_numpy() for p in basket], 1).astype(float)
    n = np.maximum(live.sum(1), 1)
    for j, p in enumerate(basket):
        W[:, close.columns.get_loc(p)] -= trig * cx.net_long * size * live[:, j] / n
    g = np.abs(W).sum(1, keepdims=True)
    return np.nan_to_num(W * np.where(g > 1.0, 1.0 / np.maximum(g, 1e-12), 1.0))


starts = sx.starts; st = close.index[starts]
dd = np.array([(btc.iloc[s:s + 337] / btc.iloc[s:s + 337].cummax() - 1).min() for s in starts])
crash = dd <= -0.15
y26 = (st >= pd.Timestamp("2026-01-01", tz="UTC"))


def stats(W):
    eq, _ = simulate(W, close, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
    r, s = eq[:, -1] / eq[:, 0] - 1, contest_score(eq)
    out = {}
    for per, m in (("2024-25", ~y26), ("2026", y26), ("all", np.ones(len(r), bool))):
        c, n = m & crash, m & ~crash
        out[per] = dict(score=np.median(s[m]), crash_mean=r[c].mean(), crash_med=np.median(r[c]), crash_worst=r[c].min(),
                        normal_med=np.median(r[n]), above=(r[m] > -0.01).mean(), worst=r[m].min(), n_crash=int(c.sum()))
    return out, r


if __name__ == "__main__":
    base, base_r = stats(cx.variants["crypto only (live book)"])
    COIN_MSTR, PLUS_CRCL = ["COINB/USD", "MSTRB/USD"], ["COINB/USD", "MSTRB/USD", "CRCLB/USD"]
    res = {}
    for kind, hourly, hold24, bname in itertools.product("ABCD", (False, True), (False, True), ("COIN+MSTR", "+CRCL")):
        tr = trigger(kind, hourly, hold24)
        name = f"{kind} {'hourly' if hourly else '00/12 '} {'hold24' if hold24 else 'nohold'} {bname}"
        res[name] = (stats(hedged(tr, COIN_MSTR if bname == "COIN+MSTR" else PLUS_CRCL)), tr.mean())
    b = base["2024-25"]
    print(f"crash windows: 2024-25 {base['2024-25']['n_crash']}, 2026 {base['2026']['n_crash']} (BTC falls >= 15% inside the window)\n")
    print(f"{'2024-25 (selection period)':34s}{'on':>6s}{'score':>7s}{'crash mean':>11s}{'crash worst':>12s}{'normal med':>11s}{'>-1%':>6s}  eligible")
    print(f"{'crypto only':34s}{'':6s}{b['score']:7.3f}{b['crash_mean']:+11.2%}{b['crash_worst']:+12.1%}{b['normal_med']:+11.2%}{b['above']:6.0%}")
    eligible = {}
    for name, ((o, _), on) in res.items():
        x = o["2024-25"]
        ok = x["score"] >= b["score"] - 0.003 and x["normal_med"] >= b["normal_med"] - 0.0005
        if ok: eligible[name] = x["crash_mean"]
        print(f"{name:34s}{on:6.1%}{x['score']:7.3f}{x['crash_mean']:+11.2%}{x['crash_worst']:+12.1%}{x['normal_med']:+11.2%}{x['above']:6.0%}  {'yes' if ok else ''}")
    pick = max(eligible, key=eligible.get) if eligible else None
    current = "A 00/12  nohold COIN+MSTR"
    print(f"\nselected on 2024-25: {pick}   ({len(eligible)} of {len(res)} eligible)")
    print(f"\n{'2026 (unseen)':34s}{'score':>7s}{'crash mean':>11s}{'crash med':>10s}{'crash worst':>12s}{'normal med':>11s}{'>-1%':>6s}{'worst':>7s}")
    for label, o in (("crypto only", base), (f"current cushion ({current})", res[current][0][0]), (f"selected ({pick})", res[pick][0][0] if pick else base)):
        x = o["2026"]
        print(f"{label[:34]:34s}{x['score']:7.3f}{x['crash_mean']:+11.2%}{x['crash_med']:+10.2%}{x['crash_worst']:+12.1%}{x['normal_med']:+11.2%}{x['above']:6.0%}{x['worst']:+7.1%}")
    # crash by crash: windows whose inside BTC fall >= 15%, grouped by the month of the fall
    print("\ncrash by crash (mean 14-day return of windows containing each >= 15% BTC fall):")
    _, pick_r = res[pick][0] if pick else (None, base_r)
    lows = pd.Series([close.index[s + int(np.argmin(btc.iloc[s:s + 337].to_numpy() / np.maximum.accumulate(btc.iloc[s:s + 337].to_numpy())))] for s in starts])
    ep = pd.Series(np.where(crash, lows.dt.strftime("%Y-%m"), ""), index=range(len(starts)))
    print(f"{'episode':10s}{'windows':>8s}{'crypto only':>12s}{'with cushion':>13s}")
    for m in sorted(set(ep[ep != ""])):
        k = (ep == m).to_numpy()
        print(f"{m:10s}{k.sum():8d}{base_r[k].mean():+12.2%}{pick_r[k].mean():+13.2%}")
