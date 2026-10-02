"""Sliding-window rate limiter for the Roostoo client. BUILD_SPEC.md Section 4.5."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """At most `rate_per_min` requests in ANY rolling `window_s` seconds, orders paced >=
    `min_order_gap_s` apart.

    A token bucket with capacity = rate lets a full bucket and its refill through in the same minute
    (45 calls in 60 s were logged when the book opened from flat on the mock account, against a 30 per
    minute exchange limit). Here the send time of every request is kept and a new one waits until fewer
    than the limit lie inside the window, so the cap holds for every 60-second span, bursts included.

    After 3 consecutive failures (via `note_failure`), the limit is halved for `penalty_s` seconds.
    """

    def __init__(
        self,
        rate_per_min: int = 30,
        min_order_gap_s: float = 1.0,
        penalty_s: float = 600.0,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_rate_per_min = rate_per_min
        self.min_order_gap_s = min_order_gap_s
        self.penalty_s = penalty_s
        self.window_s = window_s
        self._clock = clock  # injectable so tests run on a simulated clock
        self._sleep = sleep
        self._sent: deque[float] = deque()  # send times of the requests still inside the window
        self._last_order_ts = float("-inf")
        self._consecutive_failures = 0
        self._penalty_until = float("-inf")
        self._lock = threading.Lock()

    def _limit(self) -> int:
        """Requests allowed per window right now (halved while a failure penalty is active)."""
        if self._clock() < self._penalty_until:
            return max(1, self.base_rate_per_min // 2)
        return max(1, self.base_rate_per_min)

    def acquire(self, order: bool = False) -> None:
        """Block until a request (and, for orders, the minimum order gap) is allowed."""
        with self._lock:
            if order:
                gap = self.min_order_gap_s - (self._clock() - self._last_order_ts)
                if gap > 0:
                    self._sleep(gap)
            while True:
                now = self._clock()
                while self._sent and now - self._sent[0] >= self.window_s:
                    self._sent.popleft()  # left the window
                limit = self._limit()
                if len(self._sent) < limit:
                    break
                # wait for the request that must leave the window before one more fits
                wait = self._sent[len(self._sent) - limit] + self.window_s - now
                self._sleep(max(wait, 0.01))
            now = self._clock()
            self._sent.append(now)
            if order:
                self._last_order_ts = now

    def note_success(self) -> None:
        self._consecutive_failures = 0

    def note_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._penalty_until = self._clock() + self.penalty_s
