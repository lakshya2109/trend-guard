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
    # Covariance shrinkage (local research experiment, strategy/book.py's _cov_scale docstring):
    # 0 = pure sample covariance (the original, unmodified behaviour), 1 = pure constant-
    # correlation (the momentum sleeve's own, already-used model). The trend sleeve can hold
    # several dozen names against only cov_hours=168 observations -- a classic small-sample
    # regime where the raw sample covariance is noisy. Off (0.0) by default.
    cov_shrink: float = 0.0
    cov_shrink_rho: float = 0.7
    # Recency-weighting for the same covariance (hours, 0 = disabled -> equal-weighted, as
    # before). A longer cov_hours window stabilises the correlation structure but, equal-
    # weighted, also reacts more slowly to a genuine recent rise in volatility -- this lets
    # sizing track a fresh vol pickup within a long window instead of averaging it away, the
    # same idea the momentum sleeve's own vol estimate already uses (ewma_lambda).
    cov_ewma_halflife_h: float = 0.0


class MomentumConfig(BaseModel):
    """Cross-sectional long/short momentum sleeve (strategy/book.py). Members = lookback_sets x skips."""

    lookback_sets: list[list[int]] = Field(default_factory=lambda: [[72, 168, 336], [168, 336, 720]])
    skips: list[int] = Field(default_factory=lambda: [0, 12])  # hours of the latest move to ignore
    k: int = 6  # names per side
    buffer: int = 12  # a holding is kept while ranked within this many from its end
    short_share: float = 0.4  # share of gross in the short leg when both legs exist
    vol_weight_power: float = (
        0.0  # within each side weight ~ 1/vol^power (0 = equal weight, 0.5 = sqrt inverse-vol)
    )
    target_vol_daily: float = 0.015
    avg_corr: float = 0.7  # constant-correlation risk model
    gross_cap: float = 0.9
    select_hour_utc: int = 0
    select_every_h: int = 24  # re-rank every N hours, counted from select_hour_utc (24 = once a day)
    ewma_lambda: float = 0.99
    min_liquidity_usd: float = 5e6
    min_age_h: int = 720
    exclude: list[str] = Field(default_factory=lambda: ["PAXG/USD"])
    # no NEW shorts on a coin already down more than this over the lookback (crashed losers bounce);
    # None disables the filter. Existing shorts are not forced out.
    short_crash_return: float | None = -0.40
    short_crash_lookback_h: int = 1440
    # Derivatives crowding filter (BUILD_SPEC.md Section 4.5/5.6.5, roostoo_hackathon_research_report.md
    # -- never wired into v2 before this experiment, though the funding/OI download pipeline in
    # data/binance_vision.py was already built). Off by default so every existing config/backtest is
    # byte-for-byte unaffected unless explicitly enabled. When on: a long already extended
    # (sc > crowd_price_extended) with crowded, rising-OI positive funding (funding_z >
    # crowd_funding_z and oi_chg_z > crowd_oi_change_z) is a long-squeeze-down risk; a short paying
    # deeply negative funding (funding_z < -crowd_funding_z) is a short-squeeze-up risk. Either gets
    # its weight multiplied by crowd_mult, continuously, every hour (not just at selection) -- the
    # same shape as the existing overlays in book_components.
    crowd_enabled: bool = False
    crowd_funding_z: float = 2.0
    crowd_oi_change_z: float = 2.0
    crowd_price_extended: float = 1.5
    crowd_mult: float = 0.6
    crowd_lookback_days: int = 30
    # Cross-sectional ML filter (STRATEGY.md Section 3c, "the ML model that was not adopted" --
    # documented as a real, walk-forward-validated result, never actually implemented: no model
    # code exists in this repo, checked). This is a deliberately smaller reproduction (ridge, not
    # LightGBM; ~6 rank features, not their unspecified 17) of the same idea: only take a momentum
    # candidate the ML model also likes (predicted relative next-day return agrees in sign, and
    # clears ml_min_pred). Off by default; strategy/ml_filter.py has the full walk-forward/embargo
    # detail and the exact simplifications made.
    ml_enabled: bool = False
    ml_min_pred: float = 0.0
    ml_refit_days: int = 30
    ml_embargo_days: int = 2
    ml_train_window_days: int = 180
    ml_horizon_days: int = 1
    ml_l2: float = 1.0


class TeamConfig(BaseModel):
    """Teammates' combined book (strategy/team, vendored unmodified): Gunek-945's trend book and
    swyam-rd's momentum book, split as in their roostoo-last-supper config."""

    trend_share: float = 0.7
    momentum_share: float = 0.3
    momentum_vol_target: float = 0.015  # their [combined] override of the momentum book's 3%/day


class CrashHedgeConfig(BaseModel):
    """Crash cushion (ported from teammate branch swyam/bstocks-hedge, local research re-test against
    the cov_hours=336 book): short crypto-linked stock tokens against the book's net long crypto while
    BTC is falling fast. Checked only at decision hours (hourly checks whipsaw), off as soon as the
    trigger clears. Needs /v6 shorts on stock pairs confirmed on the account before enabling live."""

    enabled: bool = False
    instruments: list[str] = Field(default_factory=lambda: ["COINB/USD", "MSTRB/USD"])
    trigger_72h: float = -0.06
    trigger_24h: float = -0.04
    size: float = 0.75
    check_hours_utc: list[int] = Field(default_factory=lambda: [0, 12])
    reference: str = "BTC/USD"


class JumpGuardConfig(BaseModel):
    """No new or larger longs in a coin for `hold_h` hours after a 1-hour rise above `threshold`
    (research/paired.py: pumps of the 2025-26 listings mean-revert; held positions are not sold)."""

    enabled: bool = False
    threshold: float = 0.10  # simple return of one hourly bar
    hold_h: int = 24


class BearModeConfig(BaseModel):
    """Regime switch (research/fast_adapt.py, STRATEGY.md 19): while the equal-weight market of liquid coins
    is down more than `enter` over `lookback_h` hours, run only the v2 momentum long/short sleeve re-ranked
    every `select_every_h` hours; switch back once that return is above `exit` (hysteresis)."""

    enabled: bool = False
    lookback_h: int = 120
    enter: float = -0.08
    exit: float = -0.02
    min_liquidity_usd: float = 5e6  # coins in the market average: trailing 7-day mean daily volume
    select_every_h: int = 6  # momentum re-rank interval when book == "momentum"
    book: str = "momentum"  # bear-mode book: "momentum" (v2 momentum only) or "rebound" (oversold longs)
    rebound_k: int = 10  # rebound: hold the k liquid coins with the worst return over rebound_lookback_h
    rebound_lookback_h: int = 72
    rebound_hour_utc: int = 0  # re-picked once a day at this hour
    rebound_min_age_h: int = 720
    rebound_exclude: list[str] = Field(default_factory=lambda: ["PAXG/USD"])


class NetHedgeConfig(BaseModel):
    """Short a basket of the most liquid coins against part of the book's net exposure, so a fall in the
    whole altcoin market moves the account less (STRATEGY.md round 25)."""

    enabled: bool = False
    frac: float = 0.5  # share of the book's net exposure (longs minus shorts) hedged
    n_coins: int = 10  # basket: the n most liquid coins by trailing 7-day quote volume, equal weight
    min_liquidity_usd: float = 5_000_000  # basket members trade at least this much a day
    min_age_h: int = 720  # and have at least 30 days of history
    exclude: list[str] = Field(default_factory=lambda: ["PAXG/USD"])  # tokenized gold is not crypto beta


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
    crash_hedge: CrashHedgeConfig = Field(default_factory=CrashHedgeConfig)
    jump_guard: JumpGuardConfig = Field(default_factory=JumpGuardConfig)
    bear_mode: BearModeConfig = Field(default_factory=BearModeConfig)
    net_hedge: NetHedgeConfig = Field(default_factory=NetHedgeConfig)


class BreakerConfig(BaseModel):
    """Catastrophe circuit breaker on account drawdown (a safety net, not a trading rule: tighter
    drawdown brakes lowered the score in backtests by selling dips before rebounds)."""

    dd_level: float = 0.10
    dd_mult: float = 0.5
    dd_recover: float = 0.05
    confirm_cycles: int = 1  # consecutive updates past dd_level needed to trip (a lone bad reading does not)


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
    activity_windows_utc: list[list[int]] = Field(default_factory=lambda: [[1, 3], [13, 15]])
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
