"""Token-bucket rate limiter for the Roostoo client. BUILD_SPEC.md Section 4.5."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Token bucket: `rate_per_min` requests/minute overall, orders paced >= `min_order_gap_s` apart.

    After 3 consecutive failures (via `note_failure`), the effective rate is halved for
    `penalty_s` seconds.
    """

    def __init__(self, rate_per_min: int = 30, min_order_gap_s: float = 1.0, penalty_s: float = 600.0):
        self.base_rate_per_min = rate_per_min
        self.min_order_gap_s = min_order_gap_s
        self.penalty_s = penalty_s
        self._tokens = float(rate_per_min)
        self._capacity = float(rate_per_min)
        self._last_refill = time.monotonic()
        self._last_order_ts = 0.0
        self._consecutive_failures = 0
        self._penalty_until = 0.0
        self._lock = threading.Lock()

    def _current_rate_per_min(self) -> float:
        if time.monotonic() < self._penalty_until:
            return self.base_rate_per_min / 2
        return float(self.base_rate_per_min)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        rate = self._current_rate_per_min()
        self._tokens = min(self._capacity, self._tokens + elapsed * (rate / 60.0))
        self._last_refill = now

    def acquire(self, order: bool = False) -> None:
        """Block until a token (and, for orders, the minimum order gap) is available."""
        with self._lock:
            if order:
                gap = self.min_order_gap_s - (time.monotonic() - self._last_order_ts)
                if gap > 0:
                    time.sleep(gap)
            while True:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    break
                time.sleep(0.05)
            if order:
                self._last_order_ts = time.monotonic()

    def note_success(self) -> None:
        self._consecutive_failures = 0

    def note_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._penalty_until = time.monotonic() + self.penalty_s
