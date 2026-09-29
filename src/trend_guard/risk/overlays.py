"""Risk overlays: drawdown governor, daily brake, event, endgame, crowding, tournament.

BUILD_SPEC.md Section 7. Each overlay is a pure function returning (multiplier, reason). The engine
multiplies them together in the documented order (see `combine`) and applies the result on top of
the portfolio weights, then re-applies caps and gross limits.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from trend_guard.config import EventItem, RiskConfig

# BUILD_SPEC Section 7's crowding table hardcodes "72h return < -8%" as part of the short-crowding
# rule itself (not exposed as a CrowdingConfig field, and crowding_short's signature per spec takes
# no protective_cfg). Numerically the same as protective.squeeze_prior_drop's default, but it is a
# distinct spec constant for a distinct overlay row, so it is not read from ProtectiveConfig here.
_CROWDING_SHORT_RET72H_THRESHOLD = -0.08


def _parse_event_utc(utc_str: str) -> datetime:
    """Parse an EventItem.utc ISO8601 'Z' string into a timezone-aware UTC datetime."""
    return datetime.fromisoformat(utc_str.replace("Z", "+00:00"))


def drawdown_governor(equity: float, peak_equity: float, risk_cfg: RiskConfig) -> tuple[float, str]:
    """dd_mult = clip(1 - DD/dd_max, dd_floor, 1), DD = 1 - equity/peak. BUILD_SPEC Section 7."""
    if peak_equity <= 0:
        return 1.0, "peak_equity non-positive, drawdown governor skipped"
    dd = 1.0 - equity / peak_equity
    mult = 1.0 - dd / risk_cfg.dd_max
    mult = min(max(mult, risk_cfg.dd_floor), 1.0)
    reason = f"drawdown_governor dd={dd:.4f} mult={mult:.3f}" if mult != 1.0 else "drawdown_governor no-op"
    return mult, reason


def daily_brake(pnl_since_utc_midnight_pct: float, risk_cfg: RiskConfig) -> tuple[float, str]:
    """0.5 below `soft`; 0 below `hard` (caller interprets 0 as 'no new risk, only reductions
    allowed until next UTC day'); else 1. BUILD_SPEC Section 7."""
    if pnl_since_utc_midnight_pct < risk_cfg.daily_brake.hard:
        return 0.0, f"daily_brake HARD pnl={pnl_since_utc_midnight_pct:.4f}"
    if pnl_since_utc_midnight_pct < risk_cfg.daily_brake.soft:
        return 0.5, f"daily_brake soft pnl={pnl_since_utc_midnight_pct:.4f}"
    return 1.0, "daily_brake no-op"


def event_overlay(now: datetime, events: list[EventItem], risk_cfg: RiskConfig) -> tuple[float, str]:
    """`risk_cfg.event.mult` if `now` is within [event.utc - pre_min, event.utc + post_min] for any
    event; else 1. BUILD_SPEC Section 7."""
    pre = timedelta(minutes=risk_cfg.event.pre_min)
    post = timedelta(minutes=risk_cfg.event.post_min)
    for event in events:
        event_ts = _parse_event_utc(event.utc)
        if event_ts - pre <= now <= event_ts + post:
            return risk_cfg.event.mult, f"event_overlay '{event.name}' at {event.utc}"
    return 1.0, "event_overlay no-op"


def endgame_lockin(competition_day: int, cumulative_return: float, risk_cfg: RiskConfig) -> tuple[float, str]:
    """From competition day `from_day`, if cumulative_return >= `min_return`, multiply the vol-target
    k by (1 - sigma_cut); else 1. This multiplies k (the portfolio vol-target scaler), not the raw
    signal or weight directly -- the caller applies it alongside the other multipliers in `combine`
    on the final weight, which is equivalent since k is itself a linear scalar on weight.
    BUILD_SPEC Section 7."""
    endgame = risk_cfg.endgame
    if competition_day >= endgame.from_day and cumulative_return >= endgame.min_return:
        mult = 1.0 - endgame.sigma_cut
        return mult, f"endgame_lockin day={competition_day} ret={cumulative_return:.4f} mult={mult:.3f}"
    return 1.0, "endgame_lockin no-op"


def crowding_long(
    funding_z: float | None, oi_chg_24h: float | None, signal: float | None, risk_cfg: RiskConfig
) -> tuple[float, str]:
    """Long-side crowding filter: funding_z > z and oi_chg_24h > oi and signal > 0.7 -> crowding.mult.
    BUILD_SPEC Section 7. Returns (1.0, reason) if disabled or any input is unavailable (None)."""
    crowding = risk_cfg.crowding
    if not crowding.enabled:
        return 1.0, "crowding data unavailable"
    if funding_z is None or oi_chg_24h is None or signal is None:
        return 1.0, "crowding data unavailable"
    if funding_z > crowding.funding_z and oi_chg_24h > crowding.oi_chg_24h and signal > 0.7:
        return crowding.mult, f"crowding_long funding_z={funding_z:.2f} oi_chg={oi_chg_24h:.3f}"
    return 1.0, "crowding_long no-op"


def crowding_short(
    funding_z: float | None, ret_72h: float | None, risk_cfg: RiskConfig
) -> tuple[float, str, bool]:
    """Short-side crowding filter: funding_z < -z and 72h return < -8% (squeeze_prior_drop, from
    protective_cfg conceptually, but the trigger here is the crowding table's fixed -8%; BUILD_SPEC
    Section 7 lists this as part of the crowding row, reusing the same -8% figure documented in
    protective.squeeze_prior_drop) -> crowding.mult, and block new short entries.

    Returns (multiplier, reason, block_new_shorts). Returns (1.0, reason, False) if disabled or any
    input is unavailable (None)."""
    crowding = risk_cfg.crowding
    if not crowding.enabled:
        return 1.0, "crowding data unavailable", False
    if funding_z is None or ret_72h is None:
        return 1.0, "crowding data unavailable", False
    if funding_z < -crowding.funding_z and ret_72h < _CROWDING_SHORT_RET72H_THRESHOLD:
        return crowding.mult, f"crowding_short funding_z={funding_z:.2f} ret_72h={ret_72h:.4f}", True
    return 1.0, "crowding_short no-op", False


def tournament_overlay(
    our_return: float, twentieth_place_return: float | None, risk_cfg: RiskConfig
) -> tuple[float, str]:
    """If our_return > 20th-place regional return + buffer: above_mult (e.g. 0.8, de-risk when
    comfortably ahead); if below: below_mult (e.g. 1.15, press when behind). BUILD_SPEC Section 7.

    NOTE: stays unused while risk_cfg.tournament.enabled is False (BUILD_SPEC Section 16 unknown #6:
    organizers haven't confirmed reading the public leaderboard is allowed). The engine must not call
    this, or must ignore its output, while tournament.enabled is False.
    """
    tournament = risk_cfg.tournament
    if not tournament.enabled or twentieth_place_return is None:
        return 1.0, "tournament_overlay disabled or data unavailable"
    reason_suffix = f"our={our_return:.4f} 20th={twentieth_place_return:.4f}"
    if our_return > twentieth_place_return + tournament.buffer:
        return tournament.above_mult, f"tournament_overlay ahead {reason_suffix}"
    return tournament.below_mult, f"tournament_overlay behind {reason_suffix}"


def combine(*results: tuple[float, str]) -> tuple[float, list[str]]:
    """Multiply all overlay multipliers together and collect every non-trivial (mult != 1.0) reason.
    Matches the w_final_i = w_i * dd_mult * day_mult * event_mult * endgame_mult * tournament_mult
    * crowd_mult_i * news_mult_i * risk_scale chain in BUILD_SPEC Section 7 (the caller passes
    risk_scale in as its own (risk_scale, "risk_scale") tuple alongside the rest)."""
    total = 1.0
    reasons: list[str] = []
    for mult, reason in results:
        total *= mult
        if mult != 1.0:
            reasons.append(reason)
    return total, reasons
