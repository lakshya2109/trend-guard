# Vendored UNMODIFIED from the team repo roostoo-last-supper (bot/strategy/model.py at commit 5220b5f),
# written by teammate swyam-rd. Only this header was added. It is excluded from lint so it stays
# byte-for-byte his code and behaves exactly as he researched it.

"""The trading model: volatility-scaled cross-sectional momentum, long and short.

Pure, causal functions of hourly market data. The backtest (`run_model`) and the live
bot (`Model.step`) run the same `select_row` and `size_row`, so what we test is what trades.

Pipeline (README, "Strategy"):
  1. universe  rolling liquidity rule: trailing 7-day volume >= $5M/day, 30 days of history
  2. score     per coin, mean over 3/7/14-day lookbacks of the return (skipping the last
               12 hours, which tend to reverse) divided by the volatility expected over
               that horizon; a risk-adjusted momentum score comparable across coins
  3. select    once a day (00:00 UTC): long the 6 best positive scores, short the 6 worst
               negative scores; a holding is kept while it stays within the top/bottom 12
  4. weights   equal weight per side; 60% of gross long, 40% short when both sides exist
  5. size      every hour, scale the book to a 3%/day portfolio volatility target
               (constant-correlation model), gross capped at 0.9x equity

Backtest evidence for every choice is in README "Research log".
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class ModelParams:
    # universe
    liquidity_min_usd: float = 5e6           # trailing 7-day mean daily quote volume to be tradable
    min_age_h: int = 720                     # a coin needs 30 days of history
    exclude: tuple[str, ...] = ("PAXG/USD", "OMNI/USD", "TON/USD")
    # score
    lookbacks_h: tuple[int, ...] = (72, 168, 336)
    skip_h: int = 12                         # ignore the most recent hours (short-horizon reversal)
    ewma_lambda: float = 0.99                # volatility decay on hourly squared returns (~3-day half-life)
    # selection
    select_hours_utc: tuple[int, ...] = (0,)
    k_long: int = 6
    k_short: int = 6
    buffer_rank: int = 12                    # keep a holding while it ranks within this many from its end
    short_share: float = 0.40                # share of gross exposure in the short leg when both legs exist
    allow_short: bool = True
    # sizing
    vol_target_daily: float = 0.03
    avg_corr: float = 0.7                    # constant-correlation model for portfolio volatility
    max_gross: float = 0.9                   # longs + short collateral, fraction of equity (1x rule, with slack)

    @classmethod
    def from_toml(cls, path: str | Path, section: str = "model") -> "ModelParams":
        raw = tomllib.loads(Path(path).read_text()).get(section, {})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model params in {path}: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


# --------------------------------------------------------------------------- signals (vectorised, causal)
def daily_liquidity(quote_volume: pd.DataFrame) -> pd.DataFrame:
    """Trailing 7-day mean daily quote volume (USD)."""
    return quote_volume.rolling(7 * HOURS_PER_DAY, min_periods=3 * HOURS_PER_DAY).sum() / 7


def banded(level: pd.DataFrame, enter: float, exit_: float) -> pd.DataFrame:
    """1 once level >= enter, 0 once level < exit_, unchanged in between (causal hysteresis)."""
    raw = pd.DataFrame(np.where(level >= enter, 1.0, np.where(level < exit_, 0.0, np.nan)),
                       index=level.index, columns=level.columns)
    return raw.ffill().fillna(0.0)


def ewma_vol(close: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Hourly volatility: RiskMetrics EWMA of squared log returns."""
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, min_periods=28).mean())


def momentum_score(close: pd.DataFrame, vol: pd.DataFrame, lookbacks_h: tuple[int, ...], skip_h: int) -> pd.DataFrame:
    """Mean over lookbacks of the skipped return scaled by the volatility expected over that horizon."""
    base = close.shift(skip_h)
    return sum((base / close.shift(skip_h + L) - 1) / (vol * np.sqrt(L)) for L in lookbacks_h) / len(lookbacks_h)


@dataclass
class Signals:
    close: pd.DataFrame
    score: np.ndarray        # (T, N)
    vol_daily: np.ndarray    # (T, N) daily volatility estimate
    eligible: np.ndarray     # (T, N) bool
    index: pd.DatetimeIndex = field(repr=False)
    pairs: list[str] = field(repr=False)


def compute_signals(panel: dict[str, pd.DataFrame], p: ModelParams) -> Signals:
    close = panel["close"].drop(columns=[c for c in p.exclude if c in panel["close"].columns])
    vol = ewma_vol(close, p.ewma_lambda)
    score = momentum_score(close, vol, p.lookbacks_h, p.skip_h)
    liquid = banded(daily_liquidity(panel["quote_volume"][close.columns]), p.liquidity_min_usd, p.liquidity_min_usd) > 0
    aged = close.notna().cumsum() >= p.min_age_h
    eligible = (liquid & aged & close.notna()).to_numpy() & np.isfinite(score.to_numpy())
    return Signals(close=close, score=score.to_numpy(), vol_daily=(vol * np.sqrt(HOURS_PER_DAY)).to_numpy(),
                   eligible=eligible, index=close.index, pairs=list(close.columns))


# --------------------------------------------------------------------------- decisions (row functions, shared by backtest and live)
def _pick(order_value: np.ndarray, ok: np.ndarray, prev: list[int], k: int, buffer: int) -> list[int]:
    """Top-k of order_value among ok, keeping previous picks that are still within `buffer`."""
    ranked = np.argsort(-np.where(ok, order_value, -np.inf))
    rank = np.empty(len(ranked), dtype=int)
    rank[ranked] = np.arange(1, len(ranked) + 1)
    keep = [i for i in prev if ok[i] and rank[i] <= buffer]
    fresh = [int(i) for i in ranked[: k + len(keep)] if ok[i] and i not in keep]
    return (keep + fresh)[:k]


def select_row(score: np.ndarray, eligible: np.ndarray, prev_longs: list[int], prev_shorts: list[int],
               p: ModelParams) -> tuple[list[int], list[int]]:
    """Daily selection: indices of longs (best positive scores) and shorts (worst negative scores)."""
    longs = _pick(score, eligible & (score > 0), prev_longs, p.k_long, p.buffer_rank)
    shorts = _pick(-score, eligible & (score < 0), prev_shorts, p.k_short, p.buffer_rank) if p.allow_short else []
    return longs, shorts


def size_row(longs: list[int], shorts: list[int], vol_daily: np.ndarray, n: int, p: ModelParams) -> np.ndarray:
    """Equal weight per side, then scale to the volatility target under a constant-correlation model."""
    w = np.zeros(n)
    if longs:
        w[longs] = (1 - p.short_share if shorts else 1.0) / len(longs)
    if shorts:
        w[shorts] = -p.short_share / len(shorts)
    held = w != 0
    if not held.any():
        return w
    a = w[held] * np.nan_to_num(vol_daily[held], nan=0.05)
    var = (a ** 2).sum() + p.avg_corr * (a.sum() ** 2 - (a ** 2).sum())
    scale = min(p.vol_target_daily / np.sqrt(max(var, 1e-12)), p.max_gross / np.abs(w).sum())
    return w * scale


def run_model(sig: Signals, p: ModelParams) -> np.ndarray:
    """Target weights (T, N) for every hour: select at selection hours, resize every hour."""
    T, N = sig.score.shape
    out = np.zeros((T, N))
    longs: list[int] = []
    shorts: list[int] = []
    for t in range(T):
        if sig.index[t].hour in p.select_hours_utc:
            longs, shorts = select_row(sig.score[t], sig.eligible[t], longs, shorts, p)
        out[t] = size_row(longs, shorts, sig.vol_daily[t], N, p)
    return out


class Model:
    """Live wrapper: holds the current selection between selection hours (persisted by the bot)."""

    def __init__(self, p: ModelParams):
        self.p = p
        self.longs: list[str] = []
        self.shorts: list[str] = []
        self.last_selection: str | None = None

    def step(self, sig: Signals, force_select: bool = False) -> dict[str, float]:
        """Target weights for the latest closed hour."""
        t = len(sig.index) - 1
        now = sig.index[t]
        idx = {pair: i for i, pair in enumerate(sig.pairs)}
        stamp = now.strftime("%Y-%m-%dT%H")
        if force_select or (now.hour in self.p.select_hours_utc and self.last_selection != stamp):
            longs, shorts = select_row(sig.score[t], sig.eligible[t], [idx[x] for x in self.longs if x in idx],
                                       [idx[x] for x in self.shorts if x in idx], self.p)
            self.longs = [sig.pairs[i] for i in longs]
            self.shorts = [sig.pairs[i] for i in shorts]
            self.last_selection = stamp
        w = size_row([idx[x] for x in self.longs if x in idx], [idx[x] for x in self.shorts if x in idx],
                     sig.vol_daily[t], len(sig.pairs), self.p)
        return {pair: float(x) for pair, x in zip(sig.pairs, w) if x != 0}

    def state(self) -> dict:
        return {"longs": self.longs, "shorts": self.shorts, "last_selection": self.last_selection}

    def load_state(self, s: dict) -> None:
        self.longs, self.shorts = list(s.get("longs", [])), list(s.get("shorts", []))
        self.last_selection = s.get("last_selection")
