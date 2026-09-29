"""Tests for trend_guard.risk.news_llm: the deterministic mapping function (Section 9)."""

from __future__ import annotations

from trend_guard.config import (
    CrowdingConfig,
    DailyBrakeConfig,
    EndgameConfig,
    EventOverlayConfig,
    NewsLLMConfig,
    RiskConfig,
    TournamentConfig,
)
from trend_guard.risk.news_llm import apply_news_action


def _risk_cfg(news_enabled: bool) -> RiskConfig:
    return RiskConfig(
        dd_max=0.05,
        dd_floor=0.25,
        daily_brake=DailyBrakeConfig(soft=-0.008, hard=-0.015),
        event=EventOverlayConfig(pre_min=30, post_min=60, mult=0.5),
        endgame=EndgameConfig(from_day=11, min_return=0.04, sigma_cut=0.35),
        crowding=CrowdingConfig(enabled=True, funding_z=2.0, oi_chg_24h=0.10, mult=0.6),
        tournament=TournamentConfig(enabled=False, buffer=0.01, above_mult=0.8, below_mult=1.15),
        news_llm=NewsLLMConfig(enabled=news_enabled),
    )


def test_market_severe_high_confidence_triggers_mult():
    parsed = {
        "scope": "market",
        "coins": [],
        "severity": 3,
        "category": "macro",
        "confidence": 0.85,
        "rationale": "broad market shock",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action["news_mult"] == 0.5
    assert action["duration_h"] == 6
    assert action["scope"] == "market"


def test_market_severe_low_confidence_is_no_op():
    parsed = {
        "scope": "market",
        "coins": [],
        "severity": 3,
        "category": "macro",
        "confidence": 0.5,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action == {"news_mult": 1.0}


def test_coin_severe_hack_triggers_target_zero():
    parsed = {
        "scope": "coin",
        "coins": ["XYZ"],
        "severity": 3,
        "category": "hack",
        "confidence": 0.9,
        "rationale": "exchange hacked",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action["target_zero"] is True
    assert action["block_h"] == 24
    assert action["coins"] == ["XYZ"]


def test_coin_severe_delisting_triggers_target_zero():
    parsed = {
        "scope": "coin",
        "coins": ["ABC"],
        "severity": 3,
        "category": "delisting",
        "confidence": 0.8,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action["target_zero"] is True


def test_coin_severe_but_non_hack_category_is_no_op():
    parsed = {
        "scope": "coin",
        "coins": ["ABC"],
        "severity": 3,
        "category": "regulatory",
        "confidence": 0.9,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action == {"news_mult": 1.0}


def test_coin_low_severity_is_no_op():
    parsed = {
        "scope": "coin",
        "coins": ["ABC"],
        "severity": 1,
        "category": "hack",
        "confidence": 0.9,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action == {"news_mult": 1.0}


def test_scope_none_is_no_op():
    parsed = {
        "scope": "none",
        "coins": [],
        "severity": 0,
        "category": "other",
        "confidence": 0.0,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(True))
    assert action == {"news_mult": 1.0}


def test_disabled_config_always_no_op_even_for_severe_market_news():
    parsed = {
        "scope": "market",
        "coins": [],
        "severity": 3,
        "category": "macro",
        "confidence": 0.99,
        "rationale": "...",
    }
    action = apply_news_action(parsed, _risk_cfg(False))
    assert action == {"news_mult": 1.0}
