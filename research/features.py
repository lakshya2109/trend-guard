"""Cross-sectional feature panel for the alpha research (pure, causal: row t uses bars closed <= t).

Every feature is a (T, N) array aligned to the hourly price panel. `build` returns a dict of them plus
the forward-return labels used for training (labels look ahead by construction and are never features).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H = 24
ROOT = Path(__file__).resolve().parents[1]
EXTRA = ROOT / "data" / "extra"


def _extra(kind: str, cols: list[str], index: pd.DatetimeIndex, field: str) -> pd.DataFrame:
    """Load data/extra/<kind>/<SYM>.parquet field for each pair onto the panel's hourly index."""
    out = pd.DataFrame(np.nan, index=index, columns=cols)
    for pair in cols:
        f = EXTRA / kind / f"{pair.split('/')[0]}USDT.parquet"
        if not f.exists():
            continue
        try:
            s = pd.read_parquet(f)[field]
        except Exception:  # file still being written by the downloader
            continue
        if kind == "funding":  # settlement at t is known at t: forward-fill onto the hour
            s.index = s.index.floor("h")
            s = s[~s.index.duplicated(keep="last")]
            out[pair] = s.reindex(index.union(s.index)).ffill(limit=12).reindex(index)
        else:  # hourly bars are stamped by open time: shift to close time like the price panel
            out[pair] = s.reindex(index)
    return out


def _ts_z(x: pd.DataFrame, win: int) -> pd.DataFrame:
    m = x.rolling(win, min_periods=win // 3).mean()
    s = x.rolling(win, min_periods=win // 3).std()
    return (x - m) / s.replace(0, np.nan)


def build(close: pd.DataFrame, qv: pd.DataFrame, liq_min: float = 5e6, min_age: int = 720) -> dict:
    """Features + labels + eligibility mask."""
    px = close.ffill(limit=3)
    lp = np.log(px)
    r = lp.diff()
    vol = r.rolling(168, min_periods=72).std()  # hourly vol, 7 days
    vol_d = vol * np.sqrt(H)
    liq = qv.rolling(7 * H, min_periods=3 * H).sum() / 7
    age = close.notna().cumsum()
    elig = (liq >= liq_min) & (age >= min_age) & px.notna() & vol.notna()
    elig["PAXG/USD"] = False  # gold is not in the crypto cross-section

    # equal-weight market of eligible coins, and each coin's beta / residual against it
    mkt = r.where(elig).mean(axis=1)
    cov = r.rolling(720, min_periods=240).cov(mkt)
    beta = cov.div(mkt.rolling(720, min_periods=240).var(), axis=0)
    resid = r - beta.mul(mkt, axis=0)

    f: dict[str, pd.DataFrame] = {}
    for L in (1, 4, 12, 24, 72, 168, 336, 720):
        f[f"mom_{L}"] = (lp - lp.shift(L)) / (vol * np.sqrt(L))
    for L in (24, 168, 720):
        f[f"res_{L}"] = resid.rolling(L, min_periods=L // 2).sum() / (vol * np.sqrt(L))
    f["vol_24"] = np.log(r.rolling(24, min_periods=12).std())
    f["vol_168"] = np.log(vol)
    f["vol_ratio"] = f["vol_24"] - f["vol_168"]
    f["beta"] = beta
    f["idio_vol"] = np.log(resid.rolling(720, min_periods=240).std())
    f["max_168"] = r.rolling(168, min_periods=72).max() / vol
    f["min_168"] = r.rolling(168, min_periods=72).min() / vol
    f["skew_168"] = r.rolling(168, min_periods=72).skew()
    f["hi_720"] = (lp - lp.rolling(720, min_periods=240).max()) / (vol * np.sqrt(720))
    f["lo_720"] = (lp - lp.rolling(720, min_periods=240).min()) / (vol * np.sqrt(720))
    f["liq"] = np.log(liq)
    f["vsurge_24"] = np.log(qv.rolling(24, min_periods=12).sum() / 1 + 1) - np.log(liq + 1)
    f["vz_168"] = _ts_z(np.log(qv + 1), 720).rolling(24, min_periods=12).mean()
    f["age"] = np.log(age.clip(lower=1))

    # ---- order flow (spot taker-buy share of volume), when the extra data exists
    tbq = _extra("spot", list(close.columns), close.index, "taker_buy_quote")
    sqv = _extra("spot", list(close.columns), close.index, "quote_volume")
    trades = _extra("spot", list(close.columns), close.index, "trades")
    if tbq.notna().any().any():
        imb = (2 * tbq - sqv) / sqv.replace(0, np.nan)  # +1 all buys, -1 all sells
        for L in (1, 4, 24, 72):
            num = (2 * tbq - sqv).rolling(L, min_periods=max(1, L // 2)).sum()
            f[f"flow_{L}"] = num / sqv.rolling(L, min_periods=max(1, L // 2)).sum().replace(0, np.nan)
        f["flow_z"] = _ts_z(imb.rolling(24, min_periods=12).mean(), 720)
        f["tsize"] = np.log((sqv / trades.replace(0, np.nan)).rolling(24, min_periods=12).mean())
        f["tsize_z"] = _ts_z(f["tsize"], 720)
        f["trades_surge"] = np.log(trades.rolling(24, min_periods=12).sum() + 1) - np.log(
            trades.rolling(168, min_periods=72).sum() / 7 + 1
        )

    # ---- perp funding and basis
    fund = _extra("funding", list(close.columns), close.index, "rate")
    if fund.notna().any().any():
        f["fund"] = fund
        f["fund_3d"] = fund.rolling(72, min_periods=24).mean()
        f["fund_chg"] = fund - fund.shift(72)
    prem = _extra("premium", list(close.columns), close.index, "close")
    if prem.notna().any().any():
        f["prem_24"] = prem.rolling(24, min_periods=12).mean()
        f["prem_chg"] = f["prem_24"] - prem.rolling(168, min_periods=72).mean()
    fqv = _extra("fut", list(close.columns), close.index, "quote_volume")
    ftb = _extra("fut", list(close.columns), close.index, "taker_buy_quote")
    if fqv.notna().any().any():
        for L in (4, 24, 72):
            f[f"fflow_{L}"] = (2 * ftb - fqv).rolling(L, min_periods=max(1, L // 2)).sum() / fqv.rolling(
                L, min_periods=max(1, L // 2)
            ).sum().replace(0, np.nan)
        f["fut_spot"] = np.log(
            (fqv.rolling(24, min_periods=12).sum() + 1) / (qv.rolling(24, min_periods=12).sum() + 1)
        )
        f["fut_spot_z"] = _ts_z(f["fut_spot"], 720)

    # ---- futures open interest and positioning (Binance metrics, hourly last snapshot)
    oi = _extra("metrics", list(close.columns), close.index, "oi_value")
    if oi.notna().any().any():
        loi = np.log(oi.where(oi > 0)).ffill(limit=3)
        for L in (24, 72, 168):
            f[f"oi_chg_{L}"] = loi - loi.shift(L)
            # OI rising with price = new longs; OI falling with price up = short covering
            f[f"oi_x_px_{L}"] = np.sign(lp - lp.shift(L)) * f[f"oi_chg_{L}"]
        f["oi_to_vol"] = loi - np.log(qv.rolling(168, min_periods=72).sum() / 7 + 1)
        for col, name in (("top_sum_ls", "top_ls"), ("count_ls", "acct_ls"), ("taker_ls", "taker_ls")):
            v = np.log(_extra("metrics", list(close.columns), close.index, col).where(lambda x: x > 0)).ffill(
                limit=3
            )
            f[name] = v.rolling(24, min_periods=6).mean()
            f[f"{name}_z"] = _ts_z(f[name], 720)

    # ---- market state (same value for every coin: lets trees condition on the regime)
    btc = lp["BTC/USD"]
    mstate = {
        "m_btc_24": btc - btc.shift(24),
        "m_btc_168": btc - btc.shift(168),
        "m_mkt_24": mkt.rolling(24).sum(),
        "m_vol": np.log(mkt.rolling(168, min_periods=72).std()),
        "m_disp": np.log(r.where(elig).rolling(24, min_periods=12).sum().std(axis=1)),
    }
    for k, v in mstate.items():
        f[k] = pd.DataFrame(
            np.repeat(v.to_numpy()[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns
        )

    # ---- labels: forward residual-to-market return over h hours, in units of the coin's daily vol
    labels = {}
    for h in (24, 48, 72):
        fwd = lp.shift(-h) - lp
        fwd_ex = fwd.sub(fwd.where(elig).mean(axis=1), axis=0)
        labels[f"y_{h}"] = fwd_ex / (vol_d * np.sqrt(h / H))
        labels[f"raw_{h}"] = fwd_ex
        labels[f"ts_{h}"] = fwd / (vol_d * np.sqrt(h / H))  # own return, not relative: time-series target
    return {"f": f, "y": labels, "elig": elig, "vol_d": vol_d, "beta": beta}
