"""Typed config loading: config/strategy.yaml, config/symbols.yaml, and env.

Every threshold used by strategy/risk code must be read through StrategyConfig. No magic numbers
in code (BUILD_SPEC.md Section 0.5).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class CompetitionConfig(BaseModel):
    start_utc: str
    end_utc: str
    risk_scale: float = 1.0


class TrendConfig(BaseModel):
    """Long-only time-series trend sleeve (strategy/book.py). Members = lookback_sets x min_scores."""

    lookback_sets: list[list[int]] = Field(default_factory=lambda: [[168, 336, 720], [168, 336, 720, 1440]])
    min_scores: list[float] = Field(default_factory=lambda: [0.25, 0.4])
    score_cap: float = 2.0  # t-stat clipped to +-cap then scaled to [-1, 1]
    vol_window_h: int = 168  # rolling window for the hourly volatility estimate
    target_vol_daily: float = 0.004  # book volatility target, per day
    max_weight: float = 0.5
    min_weight: float = 0.01  # positions smaller than this cannot pay their round-trip cost
    gross_cap: float = 1.0
    cov_hours: int = 168  # trailing window for the covariance used by the vol target
    rebalance_hour_utc: int = 12
    min_liquidity_usd: float = 2e6  # trailing 7-day mean daily quote volume
    min_age_h: int = 336
    exclude: list[str] = Field(default_factory=list)


class MomentumConfig(BaseModel):
    """Cross-sectional long/short momentum sleeve (strategy/book.py). Members = lookback_sets x skips."""

    lookback_sets: list[list[int]] = Field(default_factory=lambda: [[72, 168, 336], [168, 336, 720]])
    skips: list[int] = Field(default_factory=lambda: [0, 12])  # hours of the latest move to ignore
    k: int = 6  # names per side
    buffer: int = 12  # a holding is kept while ranked within this many from its end
    short_share: float = 0.4  # share of gross in the short leg when both legs exist
    target_vol_daily: float = 0.015
    avg_corr: float = 0.7  # constant-correlation risk model
    gross_cap: float = 0.9
    select_hour_utc: int = 0
    ewma_lambda: float = 0.99
    min_liquidity_usd: float = 5e6
    min_age_h: int = 720
    exclude: list[str] = Field(default_factory=lambda: ["PAXG/USD"])
    # no NEW shorts on a coin already down more than this over the lookback (crashed losers bounce);
    # None disables the filter. Existing shorts are not forced out.
    short_crash_return: float | None = -0.40
    short_crash_lookback_h: int = 1440


class TeamConfig(BaseModel):
    """Teammates' combined book (strategy/team, vendored unmodified): Gunek-945's trend book and
    swyam-rd's momentum book, split as in their roostoo-last-supper config."""

    trend_share: float = 0.7
    momentum_share: float = 0.3
    momentum_vol_target: float = 0.015  # their [combined] override of the momentum book's 3%/day


class BookConfig(BaseModel):
    """Account = v2_share x (trend-guard v2 sleeves) + team_share x (teammates' book).

    trend_share / momentum_share split the v2 part. Every level's shares sum to <= 1 and every sleeve
    caps its own gross, so longs + short collateral can never exceed equity (no leverage)."""

    v2_share: float = 0.7
    team_share: float = 0.3
    trend_share: float = 0.7
    momentum_share: float = 0.3
    allow_short: bool = True
    risk_multiplier: float = 1.0  # scales the whole book; gross is then capped at max_gross
    max_gross: float = 1.0  # long notional + short collateral <= this share of equity (no leverage)
    team: TeamConfig = Field(default_factory=TeamConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)


class BreakerConfig(BaseModel):
    """Catastrophe circuit breaker on account drawdown (a safety net, not a trading rule: tighter
    drawdown brakes lowered the score in backtests by selling dips before rebounds)."""

    dd_level: float = 0.10
    dd_mult: float = 0.5
    dd_recover: float = 0.05


class ExecutionConfig(BaseModel):
    rebalance_minute: int = 3  # run the hourly cycle 3 minutes after the Binance candle closes
    decision_hours_utc: list[int] = Field(default_factory=lambda: [0, 12])  # also rebalance drift here
    min_trade_weight: float = 0.01  # skip resizes smaller than this share of equity...
    min_trade_frac: float = 0.25  # ...or smaller than this fraction of the target (opens/closes always)
    cash_buffer: float = 0.005  # keep this share of equity in cash for fees
    limit_reprice_s: int = 60
    limit_max_tries: int = 4
    ticker_poll_s: int = 30
    requests_per_min: int = 30
    min_order_gap_s: float = 1.0
    history_hours: int = 2400  # hourly bars loaded per pair for the live signals (100 days)
    # activity rule (contest: >= 8 active days): if no fill since a window opened, force one small trade
    activity_windows_utc: list[list[int]] = Field(default_factory=lambda: [[0, 3], [12, 15]])
    activity_min_frac: float = 0.001  # smallest activity trade, as a share of equity (~$100 on $100k)
    activity_fallback_pair: str = "BTC/USD"  # only if the book holds and targets nothing at all


class BacktestConfig(BaseModel):
    first_window: str = "2024-02-15"  # windows start daily at window_hour_utc between these dates
    last_window: str = "2026-09-15"
    window_hour_utc: int = 12
    window_days: int = 14
    fee: float = 0.001  # taker fee per side (also the flat short fee)
    slippage: float = 0.0005  # per side, on every fill
    capital: float = 100_000.0
    # SimBroker (paper mode) fee model
    maker_fee: float = 0.0005
    taker_fee: float = 0.0010
    short_fee: float = 0.0010
    slippage_bps: float = 5


class StrategyConfig(BaseModel):
    """Root config model loaded from config/strategy.yaml."""

    competition: CompetitionConfig
    book: BookConfig
    breaker: BreakerConfig
    execution: ExecutionConfig
    backtest: BacktestConfig


def load_strategy_config(path: str | Path = "config/strategy.yaml") -> StrategyConfig:
    """Load and validate config/strategy.yaml. Raises on any missing/invalid field."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return StrategyConfig.model_validate(raw)


def config_hash(path: str | Path = "config/strategy.yaml") -> str:
    """Short SHA256 hash of the raw config file, logged at startup to tie a run to a config version."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class SymbolsConfig(BaseModel):
    overrides: dict[str, str] = Field(default_factory=dict)  # roostoo pair -> binance symbol
    excluded: list[str] = Field(default_factory=list)  # never traded: stablecoins, no Binance feed


def load_symbols(path: str | Path = "config/symbols.yaml") -> SymbolsConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SymbolsConfig.model_validate(raw)


class EnvConfig(BaseModel):
    """Environment-derived settings. Never log or print the key fields in full."""

    roostoo_api_key: str = ""
    roostoo_secret_key: str = ""
    roostoo_base_url: str = "https://mock-api.roostoo.com"
    allow_live_orders: bool = False
    bot_mode: Literal["paper", "live"] = "paper"


def load_env(dotenv_path: str | Path | None = None) -> EnvConfig:
    """Load environment variables from .env (if present) and process env. .env never overrides
    variables already set in the process environment."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return EnvConfig(
        roostoo_api_key=os.environ.get("ROOSTOO_API_KEY", ""),
        roostoo_secret_key=os.environ.get("ROOSTOO_SECRET_KEY", ""),
        roostoo_base_url=os.environ.get("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"),
        allow_live_orders=os.environ.get("ALLOW_LIVE_ORDERS", "0") == "1",
        bot_mode=os.environ.get("BOT_MODE", "paper"),  # type: ignore[arg-type]
    )
