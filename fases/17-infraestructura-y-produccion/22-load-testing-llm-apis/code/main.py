"""
Lección: 22-load-testing-llm-apis
Fase: 17
Load testing LLM APIs: concurrent users,
RPS, latency p50/p95/p99, error rate,
throughput saturation, ramping, soak
testing.
"""
from __future__ import annotations
import time
import random
import math


class LoadTest:
    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.requests = 0

    def simulate_request(self, base_latency=0.5, error_rate=0.01, jitter=0.1):
        self.requests += 1
        if random.random() < error_rate:
            self.errors += 1
            return None
        latency = base_latency + random.uniform(-jitter, jitter)
        time.sleep(0.001)
        self.latencies.append(latency)
        return latency

    def percentile(self, p):
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = min(int(len(sorted_lat) * p / 100), len(sorted_lat) - 1)
        return sorted_lat[idx]

    def avg_latency(self):
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def error_rate(self):
        if self.requests == 0:
            return 0.0
        return self.errors / self.requests

    def throughput_rps(self, duration_seconds):
        if duration_seconds == 0:
            return 0.0
        return self.requests / duration_seconds


def ramp_load(load_test, max_concurrent=10, duration=2, base_latency=0.5):
    """Simulate ramping load from 1 to max_concurrent users."""
    start = time.time()
    for user in range(1, max_concurrent + 1):
        for _ in range(user):
            load_test.simulate_request(base_latency=base_latency)
    return time.time() - start


def main() -> int:
    random.seed(42)
    lt = LoadTest()
    ramp_load(lt, max_concurrent=5, duration=2)
    print(f"RPS: {lt.throughput_rps(2):.1f}, p99: {lt.percentile(99):.3f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())