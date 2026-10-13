"""Catastrophe circuit breaker on account drawdown.

Not a trading rule: in backtests, drawdown brakes at 4-7% lowered the contest score by selling dips
right before rebounds (docs/STRATEGY.md). This exists for bugs, outages and true crashes: once the
account is `dd_level` below its peak, every target is multiplied by `dd_mult` until the drawdown
recovers to within `dd_recover`. The same class runs in the backtest and the live engine, and the
live engine persists its state so a restart does not reset the peak. A trip is kept across restarts only
while `dd_level` is unchanged: after a config change it is judged again under the new level. With
`confirm_cycles` > 1 it trips only after that many consecutive readings past `dd_level`, so one bad equity
reading (8 Oct 2026: closed-short collateral missing from the exchange balance for a moment) cannot halve
the book.
"""

from __future__ import annotations

from dataclasses import dataclass

from trend_guard.config import BreakerConfig


@dataclass
class Breaker:
    cfg: BreakerConfig
    peak: float | None = None
    mult: float = 1.0
    level: float | None = None  # the dd_level `mult` was set under
    low: int = 0  # consecutive readings at or past dd_level

    def update(self, equity: float) -> float:
        """Record the latest equity and return the exposure multiple (1.0 or cfg.dd_mult)."""
        if self.peak is None or equity > self.peak:
            self.peak = equity
        if self.level != self.cfg.dd_level:  # level changed (or unknown): drop the old trip, judge it below
            self.mult, self.level = 1.0, self.cfg.dd_level
        dd = 1.0 - equity / self.peak if self.peak > 0 else 0.0
        if dd >= self.cfg.dd_level:
            self.low += 1
            if self.low >= self.cfg.confirm_cycles:
                self.mult = self.cfg.dd_mult
        else:
            self.low = 0
            if dd <= self.cfg.dd_recover:
                self.mult = 1.0
        return self.mult

    def state(self) -> dict:
        return {"peak": self.peak, "mult": self.mult, "level": self.level, "low": self.low}

    def load(self, s: dict) -> None:
        self.peak = s.get("peak")
        self.mult = float(s.get("mult", 1.0))
        self.level = s.get("level")  # missing in state saved before levels were recorded: re-judged once
        self.low = int(s.get("low", 0))
