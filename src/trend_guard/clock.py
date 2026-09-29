"""Clock abstraction so strategy/engine code never reads the wall clock directly."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
    def now_ms(self) -> int: ...


class RealClock:
    """Wall clock, used in paper and live mode."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class SimClock:
    """Deterministic clock for backtests, advanced explicitly by the backtest loop."""

    def __init__(self, start: datetime):
        self._t = start

    def now(self) -> datetime:
        return self._t

    def now_ms(self) -> int:
        return int(self._t.timestamp() * 1000)

    def advance(self, delta: timedelta) -> None:
        self._t = self._t + delta

    def set(self, t: datetime) -> None:
        self._t = t
