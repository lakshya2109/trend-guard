"""Catastrophe circuit breaker on account drawdown.

Not a trading rule: in backtests, drawdown brakes at 4-7% lowered the contest score by selling dips
right before rebounds (docs/STRATEGY.md). This exists for bugs, outages and true crashes: once the
account is `dd_level` below its peak, every target is multiplied by `dd_mult` until the drawdown
recovers to within `dd_recover`. The same class runs in the backtest and the live engine, and the
live engine persists its state so a restart does not reset the peak.
"""

from __future__ import annotations

from dataclasses import dataclass

from trend_guard.config import BreakerConfig


@dataclass
class Breaker:
    cfg: BreakerConfig
    peak: float | None = None
    mult: float = 1.0

    def update(self, equity: float) -> float:
        """Record the latest equity and return the exposure multiple (1.0 or cfg.dd_mult)."""
        if self.peak is None or equity > self.peak:
            self.peak = equity
        dd = 1.0 - equity / self.peak if self.peak > 0 else 0.0
        if dd >= self.cfg.dd_level:
            self.mult = self.cfg.dd_mult
        elif dd <= self.cfg.dd_recover:
            self.mult = 1.0
        return self.mult

    def state(self) -> dict:
        return {"peak": self.peak, "mult": self.mult}

    def load(self, s: dict) -> None:
        self.peak = s.get("peak")
        self.mult = float(s.get("mult", 1.0))
