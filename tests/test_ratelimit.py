"""Rate limiter: a hard cap per rolling minute, order pacing and the failure penalty. Section 4.5."""

from trend_guard.roostoo.ratelimit import RateLimiter


class FakeTime:
    """Simulated monotonic clock: sleeping advances it, so the tests never wait."""

    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def limiter(ft: FakeTime, rate: int = 30, gap: float = 1.0) -> RateLimiter:
    return RateLimiter(rate, gap, clock=ft.now, sleep=ft.sleep)


def max_in_any_window(sent: list[float], window: float = 60.0) -> int:
    return max(sum(1 for u in sent if t <= u < t + window) for t in sent)


def test_a_burst_never_exceeds_the_rate_in_any_rolling_minute():
    ft = FakeTime()
    rl = limiter(ft, rate=30)
    sent = []
    for _ in range(100):  # the book opening from flat: far more calls wanted than one minute allows
        rl.acquire()
        sent.append(ft.t)
        ft.sleep(0.2)  # each request takes a moment
    assert max_in_any_window(sent) == 30  # the old token bucket let 45 through here
    assert sent[29] - sent[0] < 10  # the first 30 are not slowed down
    assert sent[30] - sent[0] >= 60  # the 31st waits for the first to leave the window


def test_quiet_traffic_is_never_delayed():
    ft = FakeTime()
    rl = limiter(ft, rate=30)
    for _ in range(20):  # the engine's idle loop: a handful of calls a minute
        start = ft.t
        for _ in range(5):
            rl.acquire()
        assert ft.t == start
        ft.sleep(60)


def test_orders_are_paced_by_the_minimum_gap():
    ft = FakeTime()
    rl = limiter(ft, gap=1.0)
    rl.acquire(order=True)
    first = ft.t
    rl.acquire()  # a read in between is not paced
    assert ft.t == first
    rl.acquire(order=True)
    assert ft.t - first >= 1.0


def test_three_failures_halve_the_limit_until_the_penalty_ends():
    ft = FakeTime()
    rl = RateLimiter(30, 1.0, penalty_s=600.0, clock=ft.now, sleep=ft.sleep)
    for _ in range(3):
        rl.note_failure()
    sent = []
    for _ in range(40):
        rl.acquire()
        sent.append(ft.t)
    assert max_in_any_window(sent) == 15
    rl.note_success()
    ft.sleep(600)  # penalty over: back to the full rate
    sent = []
    for _ in range(40):
        rl.acquire()
        sent.append(ft.t)
    assert max_in_any_window(sent) == 30
