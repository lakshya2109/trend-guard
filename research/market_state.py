"""Do market-wide leverage / crowding gauges predict the book's next-day return (especially the bad days)?"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.features import _extra, build  # noqa: E402
from research.harness import Lab  # noqa: E402


def gauges(lab: Lab) -> pd.DataFrame:
    """Hourly market-level state, all causal."""
    d = build(lab.close, lab.qv)
    ok = d["elig"]
    cols, idx = lab.cols, lab.idx
    lp = np.log(lab.close.ffill(limit=3))
    r = lp.diff().where(ok)
    oi = _extra("metrics", cols, idx, "oi_value").ffill(limit=3)
    fund = _extra("funding", cols, idx, "rate")
    acct = _extra("metrics", cols, idx, "count_ls").ffill(limit=3)
    top = _extra("metrics", cols, idx, "top_sum_ls").ffill(limit=3)
    tak = _extra("metrics", cols, idx, "taker_ls").ffill(limit=3)
    toi = oi.where(ok).sum(axis=1, min_count=5)
    g = pd.DataFrame(index=idx)
    g["mkt_24"] = r.mean(axis=1).rolling(24).sum()
    g["mkt_168"] = r.mean(axis=1).rolling(168).sum()
    g["btc_24"] = lp["BTC/USD"].diff(24)
    g["vol_ratio"] = np.log(r.mean(axis=1).rolling(24).std() / r.mean(axis=1).rolling(720).std())
    g["breadth_24"] = (lp.diff(24).where(ok) > 0).mean(axis=1)
    g["oi_chg_24"] = np.log(toi).diff(24)
    g["oi_chg_72"] = np.log(toi).diff(72)
    g["oi_chg_168"] = np.log(toi).diff(168)
    g["oi_vs_30d"] = np.log(toi) - np.log(toi).rolling(720).mean()
    g["fund_mean"] = fund.where(ok).mean(axis=1).rolling(24, min_periods=8).mean()
    g["fund_z"] = (g.fund_mean - g.fund_mean.rolling(720).mean()) / g.fund_mean.rolling(720).std()
    g["acct_ls"] = np.log(acct.where(ok)).mean(axis=1).rolling(24).mean()
    g["acct_ls_z"] = (g.acct_ls - g.acct_ls.rolling(720).mean()) / g.acct_ls.rolling(720).std()
    g["top_ls_z"] = (lambda x: (x - x.rolling(720).mean()) / x.rolling(720).std())(
        np.log(top.where(ok)).mean(axis=1).rolling(24).mean()
    )
    g["taker_24"] = np.log(tak.where(ok)).mean(axis=1).rolling(24).mean()
    return g


if __name__ == "__main__":
    lab = Lab()
    g = gauges(lab)
    eq = lab.continuous(lab.book)
    day = eq.index[eq.index.hour == 12]
    br = eq.reindex(day).pct_change().shift(-1)  # book return over the NEXT 24h, from the 12:00 decision
    br3 = eq.reindex(day).shift(-3) / eq.reindex(day) - 1
    G = g.reindex(day - pd.Timedelta(hours=1))  # state at the 11:00 bar close = known at 12:00
    G.index = day
    out = {}
    for c in G.columns:
        x = G[c]
        m = x.notna() & br.notna()
        q = pd.qcut(x[m], 5, labels=False)
        row = {"ic_1d": x[m].corr(br[m], method="spearman"), "ic_3d": x[m].corr(br3[m], method="spearman")}
        for k in range(5):
            row[f"Q{k + 1}"] = br[m][q == k].mean() * 100
        row["Q1 P(<-3%)"] = (br[m][q == 0] < -0.03).mean()
        row["Q5 P(<-3%)"] = (br[m][q == 4] < -0.03).mean()
        yrs = (
            x[m].groupby(x[m].index.year).apply(lambda s, y=br[m]: s.corr(y.loc[s.index], method="spearman"))
        )
        row.update({f"ic{y % 100}": v for y, v in yrs.items()})
        out[c] = row
    with pd.option_context("display.width", 220, "display.float_format", "{:+.3f}".format):
        print("book next-day return (%) by quintile of each market gauge, from 2024")
        print(pd.DataFrame(out).T)
