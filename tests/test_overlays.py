"""Tests for trend_guard.risk.overlays."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trend_guard.config import (
    CrowdingConfig,
    DailyBrakeConfig,
    EndgameConfig,
    EventItem,
    EventOverlayConfig,
    NewsLLMConfig,
    RiskConfig,
    TournamentConfig,
)
from trend_guard.risk.overlays import (
    combine,
    crowding_long,
    crowding_short,
    daily_brake,
    drawdown_governor,
    endgame_lockin,
    event_overlay,
    tournament_overlay,
)

RISK_CFG = RiskConfig(
    dd_max=0.05,
    dd_floor=0.25,
    daily_brake=DailyBrakeConfig(soft=-0.008, hard=-0.015),
    event=EventOverlayConfig(pre_min=30, post_min=60, mult=0.5),
    endgame=EndgameConfig(from_day=11, min_return=0.04, sigma_cut=0.35),
    crowding=CrowdingConfig(enabled=True, funding_z=2.0, oi_chg_24h=0.10, mult=0.6),
    tournament=TournamentConfig(enabled=False, buffer=0.01, above_mult=0.8, below_mult=1.15),
    news_llm=NewsLLMConfig(enabled=False),
)

RISK_CFG_CROWDING_DISABLED = RISK_CFG.model_copy(
    update={"crowding": CrowdingConfig(enabled=False, funding_z=2.0, oi_chg_24h=0.10, mult=0.6)}
)


# --- drawdown_governor ---------------------------------------------------------


def test_drawdown_governor_no_drawdown_is_one():
    mult, _ = drawdown_governor(equity=100.0, peak_equity=100.0, risk_cfg=RISK_CFG)
    assert mult == 1.0


def test_drawdown_governor_exactly_at_dd_max_returns_dd_floor():
    peak = 100.0
    equity = peak * (1 - RISK_CFG.dd_max)  # DD == dd_max exactly
    mult, _ = drawdown_governor(equity=equity, peak_equity=peak, risk_cfg=RISK_CFG)
    assert abs(mult - RISK_CFG.dd_floor) < 1e-9


def test_drawdown_governor_beyond_dd_max_clips_at_floor():
    peak = 100.0
    equity = peak * (1 - 2 * RISK_CFG.dd_max)  # DD double dd_max
    mult, _ = drawdown_governor(equity=equity, peak_equity=peak, risk_cfg=RISK_CFG)
    assert mult == RISK_CFG.dd_floor


def test_drawdown_governor_partial_drawdown_interpolates():
    peak = 100.0
    equity = peak * (1 - RISK_CFG.dd_max / 2)  # DD is half of dd_max
    mult, _ = drawdown_governor(equity=equity, peak_equity=peak, risk_cfg=RISK_CFG)
    assert abs(mult - 0.5) < 1e-9  # 1 - 0.5 = 0.5, above dd_floor


# --- daily_brake -----------------------------------------------------------------


def test_daily_brake_no_op_above_soft():
    mult, _ = daily_brake(-0.001, RISK_CFG)
    assert mult == 1.0


def test_daily_brake_soft_boundary_exact_is_no_op():
    # strictly "below soft" triggers; exactly at soft should not trigger (per "below" semantics)
    mult, _ = daily_brake(RISK_CFG.daily_brake.soft, RISK_CFG)
    assert mult == 1.0


def test_daily_brake_soft_triggers_just_below():
    mult, _ = daily_brake(RISK_CFG.daily_brake.soft - 0.0001, RISK_CFG)
    assert mult == 0.5


def test_daily_brake_hard_boundary_exact_is_soft_not_hard():
    mult, _ = daily_brake(RISK_CFG.daily_brake.hard, RISK_CFG)
    assert mult == 0.5


def test_daily_brake_hard_triggers_just_below():
    mult, _ = daily_brake(RISK_CFG.daily_brake.hard - 0.0001, RISK_CFG)
    assert mult == 0.0


# --- event_overlay -----------------------------------------------------------------


def test_event_overlay_outside_window_is_one():
    events = [EventItem(name="CPI", utc="2026-10-14T12:30:00Z")]
    now = datetime(2026, 10, 14, 8, 0, 0, tzinfo=UTC)  # hours before, outside pre_min
    mult, _ = event_overlay(now, events, RISK_CFG)
    assert mult == 1.0


def test_event_overlay_within_pre_window():
    events = [EventItem(name="CPI", utc="2026-10-14T12:30:00Z")]
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=UTC) - timedelta(minutes=15)
    mult, _ = event_overlay(now, events, RISK_CFG)
    assert mult == RISK_CFG.event.mult


def test_event_overlay_within_post_window():
    events = [EventItem(name="CPI", utc="2026-10-14T12:30:00Z")]
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=UTC) + timedelta(minutes=45)
    mult, _ = event_overlay(now, events, RISK_CFG)
    assert mult == RISK_CFG.event.mult


def test_event_overlay_just_outside_post_window():
    events = [EventItem(name="CPI", utc="2026-10-14T12:30:00Z")]
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=UTC) + timedelta(minutes=61)
    mult, _ = event_overlay(now, events, RISK_CFG)
    assert mult == 1.0


def test_event_overlay_no_events_is_one():
    now = datetime(2026, 10, 14, 12, 30, 0, tzinfo=UTC)
    mult, _ = event_overlay(now, [], RISK_CFG)
    assert mult == 1.0


# --- endgame_lockin -----------------------------------------------------------------


def test_endgame_lockin_before_from_day_is_no_op():
    mult, _ = endgame_lockin(competition_day=5, cumulative_return=0.10, risk_cfg=RISK_CFG)
    assert mult == 1.0


def test_endgame_lockin_on_from_day_with_return_met_triggers():
    mult, _ = endgame_lockin(competition_day=11, cumulative_return=0.04, risk_cfg=RISK_CFG)
    assert abs(mult - (1 - RISK_CFG.endgame.sigma_cut)) < 1e-9


def test_endgame_lockin_after_from_day_but_return_not_met_is_no_op():
    mult, _ = endgame_lockin(competition_day=12, cumulative_return=0.01, risk_cfg=RISK_CFG)
    assert mult == 1.0


def test_endgame_lockin_after_from_day_and_return_met_triggers():
    mult, _ = endgame_lockin(competition_day=14, cumulative_return=0.10, risk_cfg=RISK_CFG)
    assert abs(mult - (1 - RISK_CFG.endgame.sigma_cut)) < 1e-9


# --- crowding --------------------------------------------------------------------


def test_crowding_long_disabled_returns_neutral():
    mult, reason = crowding_long(3.0, 0.20, 0.9, RISK_CFG_CROWDING_DISABLED)
    assert mult == 1.0
    assert "unavailable" in reason


def test_crowding_long_none_input_returns_neutral():
    mult, reason = crowding_long(None, 0.20, 0.9, RISK_CFG)
    assert mult == 1.0
    assert "unavailable" in reason


def test_crowding_long_triggers():
    mult, _ = crowding_long(funding_z=2.5, oi_chg_24h=0.15, signal=0.8, risk_cfg=RISK_CFG)
    assert mult == RISK_CFG.crowding.mult


def test_crowding_long_no_trigger_when_signal_not_strong():
    mult, _ = crowding_long(funding_z=2.5, oi_chg_24h=0.15, signal=0.5, risk_cfg=RISK_CFG)
    assert mult == 1.0


def test_crowding_short_disabled_returns_neutral():
    mult, reason, block = crowding_short(-3.0, -0.10, RISK_CFG_CROWDING_DISABLED)
    assert mult == 1.0
    assert block is False
    assert "unavailable" in reason


def test_crowding_short_none_input_returns_neutral():
    mult, reason, block = crowding_short(None, -0.10, RISK_CFG)
    assert mult == 1.0
    assert block is False


def test_crowding_short_triggers_and_blocks():
    mult, _, block = crowding_short(funding_z=-2.5, ret_72h=-0.10, risk_cfg=RISK_CFG)
    assert mult == RISK_CFG.crowding.mult
    assert block is True


def test_crowding_short_no_trigger_when_return_not_deep_enough():
    mult, _, block = crowding_short(funding_z=-2.5, ret_72h=-0.02, risk_cfg=RISK_CFG)
    assert mult == 1.0
    assert block is False


# --- tournament_overlay --------------------------------------------------------------


def test_tournament_overlay_disabled_is_neutral():
    mult, _ = tournament_overlay(our_return=0.10, twentieth_place_return=0.05, risk_cfg=RISK_CFG)
    assert mult == 1.0


def test_tournament_overlay_enabled_ahead_applies_above_mult():
    cfg = RISK_CFG.model_copy(
        update={"tournament": TournamentConfig(enabled=True, buffer=0.01, above_mult=0.8, below_mult=1.15)}
    )
    mult, _ = tournament_overlay(our_return=0.10, twentieth_place_return=0.05, risk_cfg=cfg)
    assert mult == cfg.tournament.above_mult


def test_tournament_overlay_enabled_behind_applies_below_mult():
    cfg = RISK_CFG.model_copy(
        update={"tournament": TournamentConfig(enabled=True, buffer=0.01, above_mult=0.8, below_mult=1.15)}
    )
    mult, _ = tournament_overlay(our_return=0.01, twentieth_place_return=0.05, risk_cfg=cfg)
    assert mult == cfg.tournament.below_mult


# --- combine --------------------------------------------------------------------


def test_combine_multiplies_all_and_collects_reasons():
    total, reasons = combine((0.5, "a"), (0.6, "b"), (1.0, "c"))
    assert abs(total - 0.3) < 1e-9
    assert reasons == ["a", "b"]


def test_combine_all_neutral():
    total, reasons = combine((1.0, "a"), (1.0, "b"))
    assert total == 1.0
    assert reasons == []
