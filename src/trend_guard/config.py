"""Typed config loading: config/strategy.yaml, config/events.yaml, config/symbols.yaml, and env.

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


class UniverseConfig(BaseModel):
    max_crypto: int = 25
    max_spread_pct: float = 0.0015
    min_history_days: int = 180
    majors: list[str] = Field(default_factory=list)
    bstocks_enabled: bool = False


class SignalConfig(BaseModel):
    lookbacks_h: list[int]
    w_donchian: float = 0.5
    w_tsmom: float = 0.5
    smooth_bars: int = 3
    # "hourly": ensemble on 1h closes over lookbacks_h. "daily": the paper-style ensemble on daily
    # closes over lookbacks_d. "blend": mean of the two (BUILD_SPEC Section 11.3 sweep).
    mode: Literal["hourly", "daily", "blend"] = "hourly"
    lookbacks_d: list[int] = Field(default_factory=lambda: [5, 10, 20, 30, 60, 90])


class VolConfig(BaseModel):
    halflife_h: int = 48
    vol_floor_ann: float = 0.15


class RegimeThreshold(BaseModel):
    mkt: float
    breadth: float


class RegimeBudget(BaseModel):
    long: float
    short: float
    long_entry_min: float
    short_entry_max: float


class RegimeConfig(BaseModel):
    index_size: int = 10
    on: RegimeThreshold
    off: RegimeThreshold
    hysteresis: float = 0.10
    budgets: dict[str, RegimeBudget]
    # per-asset entry/exit band: when True, an open position is kept while its signal stays on the
    # same side of zero, even after it falls back inside the entry threshold (False = spec default:
    # entry and exit share one threshold)
    hold_to_zero: bool = False


class DiversifiersConfig(BaseModel):
    entry_min: float = 0.10
    allow_short: bool = False


class PortfolioCaps(BaseModel):
    btc_eth: float = 0.25
    major: float = 0.12
    other: float = 0.06
    bstock: float = 0.08
    bstock_sleeve: float = 0.25
    paxg: float = 0.20
    short_total: float = 0.50


class PortfolioConfig(BaseModel):
    sigma_asset_target: float = 0.15
    sigma_portfolio_target: float = 0.30
    cov_halflife_h: int = 72
    shrink: float = 0.30
    k_max: float = 3.0
    gross_max: float = 0.98
    caps: PortfolioCaps


class DailyBrakeConfig(BaseModel):
    soft: float
    hard: float


class EventOverlayConfig(BaseModel):
    pre_min: int = 30
    post_min: int = 60
    mult: float = 0.5


class EndgameConfig(BaseModel):
    from_day: int
    min_return: float
    sigma_cut: float


class CrowdingConfig(BaseModel):
    enabled: bool = True
    funding_z: float = 2.0
    oi_chg_24h: float = 0.10
    mult: float = 0.6


class TournamentConfig(BaseModel):
    enabled: bool = False
    buffer: float = 0.01
    above_mult: float = 0.8
    below_mult: float = 1.15


class NewsLLMConfig(BaseModel):
    enabled: bool = False


class RiskConfig(BaseModel):
    dd_max: float = 0.05
    dd_floor: float = 0.25
    daily_brake: DailyBrakeConfig
    event: EventOverlayConfig
    endgame: EndgameConfig
    crowding: CrowdingConfig
    tournament: TournamentConfig
    news_llm: NewsLLMConfig


class ProtectiveConfig(BaseModel):
    atr_mult: float = 2.5
    atr_bar_h: int = 1  # bar size the ATR is measured on (1 = hourly bars, spec default; 24 = daily)
    atr_window: int = 24  # number of those bars averaged
    shock_z: float = 2.5
    squeeze_prior_drop: float = -0.08
    cooldown_h: int = 6


class ExecutionConfig(BaseModel):
    rebalance_minute: int = 3
    deadband: float = 0.02
    min_hold_h: int = 3
    limit_reprice_s: int = 60
    limit_max_tries: int = 4
    ticker_poll_s: int = 30
    requests_per_min: int = 30
    min_order_gap_s: float = 1.0


class BacktestConfig(BaseModel):
    start: str = "2024-01-01"
    half_spread: float = 0.0003
    slippage_bps: float = 2
    maker_fee: float = 0.0005
    taker_fee: float = 0.0010
    short_fee: float = 0.0010
    window_days: int = 14


class StrategyConfig(BaseModel):
    """Root config model loaded from config/strategy.yaml."""

    competition: CompetitionConfig
    universe: UniverseConfig
    signal: SignalConfig
    vol: VolConfig
    regime: RegimeConfig
    diversifiers: DiversifiersConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    protective: ProtectiveConfig
    execution: ExecutionConfig
    backtest: BacktestConfig


def load_strategy_config(path: str | Path = "config/strategy.yaml") -> StrategyConfig:
    """Load and validate config/strategy.yaml. Raises on any missing/invalid field."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return StrategyConfig.model_validate(raw)


def config_hash(path: str | Path = "config/strategy.yaml") -> str:
    """Short SHA256 hash of the raw config file, logged at startup to tie a run to a config version."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class EventItem(BaseModel):
    name: str
    utc: str


def load_events(path: str | Path = "config/events.yaml") -> list[EventItem]:
    """Load the scheduled macro event calendar. Returns [] if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [EventItem.model_validate(e) for e in raw.get("events", []) or []]


class SymbolsConfig(BaseModel):
    overrides: dict[str, str] = Field(default_factory=dict)
    asset_class: dict[str, str] = Field(default_factory=dict)
    bstocks: dict[str, str] = Field(default_factory=dict)
    crypto_beta_stocks: list[str] = Field(default_factory=list)
    stablecoins: list[str] = Field(default_factory=list)


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
    anthropic_api_key: str = ""


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
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
