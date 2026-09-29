"""Portfolio construction: inverse-vol sizing, caps, covariance shrinkage, vol targeting, gross cap.

BUILD_SPEC.md Section 6.4-6.5. Pure functions only. Weights are fractions of equity: positive = long
notional, negative = short (|w| * equity is short collateral).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.config import PortfolioConfig, VolConfig
from trend_guard.strategy.signals import ANN_H

__all__ = [
    "ANN_H",
    "sigma_ann",
    "raw_weight",
    "apply_caps",
    "apply_aggregate_caps",
    "portfolio_vol",
    "shrink_covariance",
    "scale_to_target",
    "enforce_gross",
    "build_portfolio",
]

_CAP_BY_BUCKET_ATTR = {
    "btc_eth": "btc_eth",
    "major": "major",
    "other": "other",
    "bstock": "bstock",
    "paxg": "paxg",
}


def sigma_ann(sigma_hourly_series: pd.Series) -> pd.Series:
    """Annualise an hourly EWMA vol series. BUILD_SPEC Section 6.4."""
    return sigma_hourly_series * np.sqrt(ANN_H)


def raw_weight(s_eff: float, sigma_ann_i: float, vol_cfg: VolConfig, portfolio_cfg: PortfolioConfig) -> float:
    """Inverse-vol sized raw weight for one asset. BUILD_SPEC Section 6.4.

    NaN s_eff (asset gated out / insufficient history) propagates to NaN weight, which the caller
    (build_portfolio, or the engine) should treat as 0 exposure.
    """
    if s_eff != s_eff:  # NaN check
        return float("nan")
    return s_eff * portfolio_cfg.sigma_asset_target / max(sigma_ann_i, vol_cfg.vol_floor_ann)


def apply_caps(
    weights: dict[str, float], bucket_of: dict[str, str], portfolio_cfg: PortfolioConfig
) -> dict[str, float]:
    """Clip abs(weight) per asset to its bucket's cap. BUILD_SPEC Section 6.5.

    `bucket_of` maps asset -> one of "btc_eth"/"major"/"other"/"bstock"/"paxg". short_total and
    bstock_sleeve are aggregate caps, applied separately by apply_aggregate_caps. NaN weights pass
    through unchanged (caller treats NaN as 0 exposure).
    """
    capped: dict[str, float] = {}
    for asset, w in weights.items():
        if w != w:  # NaN check
            capped[asset] = w
            continue
        bucket = bucket_of.get(asset)
        attr = _CAP_BY_BUCKET_ATTR.get(bucket) if bucket else None
        cap = getattr(portfolio_cfg.caps, attr) if attr is not None else None
        capped[asset] = float(np.clip(w, -cap, cap)) if cap is not None else w
    return capped


def _scale_group_pro_rata(
    weights: dict[str, float], members: list[str], group_total: float, cap: float
) -> dict[str, float]:
    """If group_total exceeds cap, scale every member's weight down by the same factor (preserves
    relative proportions within the group)."""
    if group_total <= cap or group_total <= 0:
        return weights
    factor = cap / group_total
    scaled = dict(weights)
    for asset in members:
        w = scaled[asset]
        if w == w:  # skip NaN
            scaled[asset] = w * factor
    return scaled


def apply_aggregate_caps(
    weights: dict[str, float], bucket_of: dict[str, str], portfolio_cfg: PortfolioConfig
) -> dict[str, float]:
    """Scale down pro rata within the group if total short collateral exceeds caps.short_total, or
    total bstock sleeve exceeds caps.bstock_sleeve. BUILD_SPEC Section 6.5."""
    out = dict(weights)

    short_members = [a for a, w in out.items() if w == w and w < 0]
    short_total = sum(-out[a] for a in short_members)
    out = _scale_group_pro_rata(out, short_members, short_total, portfolio_cfg.caps.short_total)

    bstock_members = [a for a in out if bucket_of.get(a) == "bstock" and out[a] == out[a]]
    bstock_total = sum(abs(out[a]) for a in bstock_members)
    out = _scale_group_pro_rata(out, bstock_members, bstock_total, portfolio_cfg.caps.bstock_sleeve)

    return out


def portfolio_vol(weights: dict[str, float], cov: pd.DataFrame) -> float:
    """sqrt(w' Sigma w). Assets present in `weights` but absent from `cov` are treated as 0 weight
    for this computation (aligned via cov's index/columns)."""
    assets = list(cov.index)
    w = np.array([weights.get(a, 0.0) or 0.0 for a in assets], dtype=float)
    w = np.nan_to_num(w, nan=0.0)
    sigma = cov.loc[assets, assets].to_numpy()
    variance = float(w @ sigma @ w)
    return float(np.sqrt(max(variance, 0.0)))


def shrink_covariance(cov: pd.DataFrame, shrink: float) -> pd.DataFrame:
    """(1-shrink)*Sigma + shrink*diag(Sigma). BUILD_SPEC Section 6.4."""
    diag = pd.DataFrame(np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns)
    return (1 - shrink) * cov + shrink * diag


def scale_to_target(
    weights: dict[str, float], cov: pd.DataFrame, portfolio_cfg: PortfolioConfig
) -> dict[str, float]:
    """k = min(sigma_portfolio_target / max(port_vol, 1e-6), k_max); scale weights by k. Caller is
    responsible for re-applying apply_caps / apply_aggregate_caps afterward per the Section 6.4
    pseudocode order (this function only scales)."""
    port_vol = portfolio_vol(weights, cov)
    k = min(portfolio_cfg.sigma_portfolio_target / max(port_vol, 1e-6), portfolio_cfg.k_max)
    return {a: (w * k if w == w else w) for a, w in weights.items()}


def enforce_gross(weights: dict[str, float], gross_max: float) -> dict[str, float]:
    """Scale all weights down pro rata so sum(abs(weight)) == gross_max, if over budget.
    BUILD_SPEC Section 6.4/6.5."""
    gross = sum(abs(w) for w in weights.values() if w == w)
    if gross <= gross_max or gross <= 0:
        return dict(weights)
    factor = gross_max / gross
    return {a: (w * factor if w == w else w) for a, w in weights.items()}


def build_portfolio(
    s_eff: dict[str, float],
    sigma_ann_by_asset: dict[str, float],
    cov: pd.DataFrame,
    bucket_of: dict[str, str],
    vol_cfg: VolConfig,
    portfolio_cfg: PortfolioConfig,
) -> dict[str, float]:
    """Orchestrates the full chain, matching the BUILD_SPEC Section 6.4 pseudocode order exactly:
    raw_w -> caps -> Sigma_s (shrink) -> k (scale_to_target) -> caps again -> enforce_gross.
    """
    raw_w = {
        a: raw_weight(s, sigma_ann_by_asset.get(a, float("nan")), vol_cfg, portfolio_cfg)
        for a, s in s_eff.items()
    }
    w = apply_caps(raw_w, bucket_of, portfolio_cfg)
    w = apply_aggregate_caps(w, bucket_of, portfolio_cfg)

    sigma_s = shrink_covariance(cov, portfolio_cfg.shrink)
    w = scale_to_target(w, sigma_s, portfolio_cfg)

    w = apply_caps(w, bucket_of, portfolio_cfg)
    w = apply_aggregate_caps(w, bucket_of, portfolio_cfg)

    w = enforce_gross(w, portfolio_cfg.gross_max)
    return w
