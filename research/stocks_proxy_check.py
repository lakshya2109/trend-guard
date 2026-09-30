"""How well does the underlying stock (Yahoo, forward-filled outside US hours) stand in for the 24/7
Binance token? Measured over the overlap (Jun-Sep 2026): hourly returns during US trading hours,
token premium to the underlying, and 14-day returns from 12:00 UTC (what the contest scores)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((ROOT / "data/bars_stocks/1h").glob("*.parquet")):
    pf = ROOT / "data/bars_proxy/1h" / f.name
    if not pf.exists():
        continue
    tok = pd.read_parquet(f)["close"]; tok.index = pd.to_datetime(tok.index, utc=True)
    px = pd.read_parquet(pf)["close"]; px.index = pd.to_datetime(px.index, utc=True)
    idx = tok.index[tok.index >= max(tok.index[0], px.index[0]) + pd.Timedelta(days=3)]
    tok = tok.reindex(idx); open_hours = idx.intersection(px.index)
    pxf = px.reindex(px.index.union(idx)).ffill().reindex(idx)
    r_tok = tok.pct_change(); r_px = pxf.pct_change()
    live = open_hours[1:]
    corr_h = r_tok.reindex(live).corr(r_px.reindex(live))
    prem = (tok.reindex(open_hours) / px.reindex(open_hours) - 1)
    starts = [s for s in idx if s.hour == 12 and s + pd.Timedelta(days=14) in idx]
    r14_tok = np.array([tok[s + pd.Timedelta(days=14)] / tok[s] - 1 for s in starts])
    r14_px = np.array([pxf[s + pd.Timedelta(days=14)] / pxf[s] - 1 for s in starts])
    rows.append({"token": f.stem.replace("USDT", ""), "days": (idx[-1] - idx[0]).days, "corr_1h_us_hours": corr_h,
                 "premium_median": prem.median(), "premium_p95_abs": prem.abs().quantile(0.95),
                 "corr_14d": np.corrcoef(r14_tok, r14_px)[0, 1] if len(starts) > 5 else np.nan,
                 "track_err_14d": np.std(r14_tok - r14_px) if len(starts) > 5 else np.nan,
                 "tok_vol_d": r_tok.std() * np.sqrt(24), "offhours_share_of_var": (r_tok.drop(live, errors="ignore") ** 2).sum() / (r_tok ** 2).sum()})
df = pd.DataFrame(rows).set_index("token")
pd.set_option("display.width", 200)
print(df.round(4).to_string())
print(f"\nmedian across tokens: 1h corr (US hours) {df.corr_1h_us_hours.median():.2f}, 14d-return corr {df.corr_14d.median():.2f}, "
      f"14d tracking error {df.track_err_14d.median():.2%}, |premium| p95 {df.premium_p95_abs.median():.2%}, "
      f"share of token variance outside US hours {df.offhours_share_of_var.median():.0%}")
