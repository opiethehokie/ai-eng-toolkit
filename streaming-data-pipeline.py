import asyncio
import random
import time
from collections import Counter, deque

import numpy as np
from datasketch import HyperLogLog
from probables import CountMinSketch

MAX_WINDOW_SIZE = 1000

BATCH_SIZE = 100
MAX_DELAY = 0.5

# streaming structures
hll = HyperLogLog(p=8)
cms = CountMinSketch(width=1000, depth=5)
value_window = deque(maxlen=MAX_WINDOW_SIZE)
latency_window = deque(maxlen=MAX_WINDOW_SIZE)

# Welford's algorithm for rolling mean/variance
n = 0
mean = 0.0
M2 = 0.0


# push events to subscribers
async def publisher(queue: asyncio.Queue) -> None:
    while True:
        event = {
            "user_id": random.randint(1, 5000),
            "value": random.gauss(50, 10),
            "timestamp": time.time(),
        }
        await queue.put(event)  # blocks if full
        await asyncio.sleep(0.02 * random.random())  # ~100 events/sec


# calculate statistics for one batch of events
def process_batch(events: list[dict]) -> None:
    global n, mean, M2

    for event in events:
        user_id = str(event["user_id"])
        value = event["value"]
        latency = time.time() - event["timestamp"]

        hll.update(user_id.encode("utf-8"))
        cms.add(user_id)
        value_window.append(value)
        latency_window.append(latency)

        n += 1
        delta = value - mean
        mean += delta / n
        M2 += delta * (value - mean)

    print(f"\n[{time.strftime('%X')}] Processed {n} events")

    print(f"Estimated unique users: {hll.count():.0f}")
    print(f"Estimated number of events with user_id=123: {cms.check('123'):.0f}")

    top_items = Counter([round(v) for v in value_window]).most_common(3)
    print(f"Top-3 value buckets: {top_items}")
    variance = M2 / (n - 1) if n > 1 else 0
    print(f"Rolling mean: {mean:.2f}, variance: {variance:.2f}")

    latencies = np.array(latency_window)
    print(
        f"Latency (ms): p50={np.percentile(latencies, 50) * 1000:.2f}, "
        f"p95={np.percentile(latencies, 95) * 1000:.2f}, "
        f"p99={np.percentile(latencies, 99) * 1000:.2f}"
    )


# consume events from the queue until we have a full batch or enough time passes
async def subscriber(queue: asyncio.Queue) -> None:
    buffer = []
    start = time.time()

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=MAX_DELAY)
            buffer.append(event)

            if len(buffer) >= BATCH_SIZE or (time.time() - start) >= MAX_DELAY:
                process_batch(buffer)
                buffer.clear()
                start = time.time()
        except asyncio.TimeoutError:
            if buffer:
                process_batch(buffer)
                buffer.clear()
                start = time.time()


async def log_queue_size(queue: asyncio.Queue, interval=5.0) -> None:
    while True:
        print(f"\n[{time.strftime('%X')}] Queue size: {queue.qsize()} / {queue.maxsize}")
        await asyncio.sleep(interval)


async def main():
    queue = asyncio.Queue(maxsize=10)  # simulated pub-sub queue wit bounded size for backpressure
    await asyncio.gather(publisher(queue), subscriber(queue), log_queue_size(queue))


asyncio.run(main())
