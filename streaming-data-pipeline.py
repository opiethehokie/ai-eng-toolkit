import asyncio
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from datasketch import HyperLogLog
from probables import CountMinSketch

MAX_WINDOW_SIZE = 1000

BATCH_SIZE = 100
MAX_DELAY = 0.1
DASHBOARD_HISTORY = 5
PROCESSING_DELAY_RANGE = (0.02, 0.06)
P99_HISTORY_SIZE = 30
P99_STDDEV_MULTIPLIER = 2.0
P99_MIN_BASELINE_MS = 50.0


@dataclass(frozen=True)
class Event:
    """Single event in the stream."""

    user_id: int
    value: float
    timestamp: float


class Stats:
    """Streaming statistics for the pipeline."""

    def __init__(self, window_size: int) -> None:
        self.hll = HyperLogLog(p=8)
        self.cms = CountMinSketch(width=1000, depth=5)
        self.value_window = deque(maxlen=window_size)
        self.latency_window = deque(maxlen=window_size)

        # Welford's algorithm for rolling mean/variance
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, event: Event) -> None:
        """Update all streaming stats with one event."""
        user_id = str(event.user_id)
        latency = time.time() - event.timestamp

        self.hll.update(user_id.encode("utf-8"))
        self.cms.add(user_id)
        self.value_window.append(event.value)
        self.latency_window.append(latency)

        self.n += 1
        delta = event.value - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (event.value - self.mean)

    def latency_percentiles(self) -> tuple[float, float, float]:
        """Return p50/p95/p99 latencies in milliseconds."""
        if not self.latency_window:
            return (0.0, 0.0, 0.0)
        latencies = np.array(self.latency_window)
        return (
            float(np.percentile(latencies, 50) * 1000),
            float(np.percentile(latencies, 95) * 1000),
            float(np.percentile(latencies, 99) * 1000),
        )


class Dashboard:
    """Terminal dashboard that renders a compact, readable view."""

    def __init__(self, history_size: int) -> None:
        self.history = deque(maxlen=history_size)
        self.p99_history = deque(maxlen=P99_HISTORY_SIZE)

    def push(self, line: str) -> None:
        self.history.append(line)

    def record_p99(self, p99: float) -> None:
        self.p99_history.append(p99)

    def p99_alert_threshold(self) -> float | None:
        if len(self.p99_history) < max(10, P99_HISTORY_SIZE // 3):
            return None
        mean = statistics.fmean(self.p99_history)
        stddev = statistics.pstdev(self.p99_history)
        return max(P99_MIN_BASELINE_MS, mean + (P99_STDDEV_MULTIPLIER * stddev))

    def render(
        self,
        stats: Stats,
        queue: asyncio.Queue[Event],
        p50: float,
        p95: float,
        p99: float,
    ) -> None:
        # ANSI clear screen + home cursor
        print("\033[2J\033[H", end="")
        print("Streaming Data Pipeline — Live Dashboard")
        print("-" * 54)
        print(
            f"Events: {stats.n:<8} "
            f"Unique Users(est): {stats.hll.count():<8.0f} "
            f"Queue: {queue.qsize()}/{queue.maxsize}"
        )
        variance = stats.M2 / (stats.n - 1) if stats.n > 1 else 0.0
        print(f"Mean: {stats.mean:>6.2f}  Var: {variance:>7.2f}  Window: {len(stats.value_window)}/{stats.value_window.maxlen}")
        print(f"Latency ms: p50={p50:>7.2f}  p95={p95:>7.2f}  p99={p99:>7.2f}")
        print("-" * 54)
        print("Recent batches:")
        for line in self.history:
            print(f"  {line}")


async def publisher(queue: asyncio.Queue[Event]) -> None:
    """Push events into the queue at a steady rate."""
    while True:
        event = Event(
            user_id=random.randint(1, 5000),
            value=random.gauss(50, 10),
            timestamp=time.time(),
        )
        await queue.put(event)  # blocks if full
        await asyncio.sleep(0.02 * random.random())  # ~100 events/sec


def process_batch(
    events: list[Event],
    stats: Stats,
    dashboard: Dashboard,
    queue: asyncio.Queue[Event],
) -> None:
    """Update stats for a batch, then render dashboard and alerts."""
    for event in events:
        stats.update(event)
    p50, p95, p99 = stats.latency_percentiles()
    dashboard.record_p99(p99)
    dashboard.push(
        f"{time.strftime('%X')} | "
        f"n={stats.n:<6} | "
        f"p50={p50:>6.1f} p95={p95:>6.1f} p99={p99:>6.1f}"
    )
    dashboard.render(stats, queue, p50, p95, p99)
    threshold = dashboard.p99_alert_threshold()
    if threshold is not None and p99 > threshold:
        print(
            f"\nALERT: High latency p99={p99:.2f}ms "
            f"exceeds dynamic threshold {threshold:.2f}ms"
        )


async def batcher(queue: asyncio.Queue[Event], batch_size: int, max_delay: float) -> AsyncIterator[list[Event]]:
    """Yield event batches based on size or max delay."""
    buffer = []
    start = time.time()

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=max_delay)
            buffer.append(event)

            if len(buffer) >= batch_size or (time.time() - start) >= max_delay:
                yield buffer
                buffer = []
                start = time.time()
        except asyncio.TimeoutError:
            if buffer:
                yield buffer
                buffer = []
                start = time.time()


async def subscriber(
    queue: asyncio.Queue[Event],
    stats: Stats,
    dashboard: Dashboard,
) -> None:
    """Consume batches from the queue and update stats."""
    async for events in batcher(queue, BATCH_SIZE, MAX_DELAY):
        await asyncio.sleep(random.uniform(*PROCESSING_DELAY_RANGE))
        process_batch(events, stats, dashboard, queue)


async def main() -> None:
    """Run the publisher and subscriber."""
    queue = asyncio.Queue(maxsize=10)  # simulated pub-sub queue with bounded size for backpressure
    stats = Stats(window_size=MAX_WINDOW_SIZE)
    dashboard = Dashboard(history_size=DASHBOARD_HISTORY)
    await asyncio.gather(publisher(queue), subscriber(queue, stats, dashboard))


asyncio.run(main())
