"""
Lección: 22-async-hogwild-inference
Fase: 10
Async inference: async/sync, batching, Hogwild-style no coordination.
LLM serving optimization.
"""
from __future__ import annotations
import sys
import numpy as np


def async_batch_simulation(requests, batch_size=4, max_wait_ms=10):
    """Simula async batch inference.
    requests: list of (prompt, callback). batch_size: max batch.
    max_wait_ms: max wait time antes de procesar batch.
    """
    pending = []
    results = []
    for req_id, prompt in requests:
        pending.append((req_id, prompt))
        if len(pending) >= batch_size:
            # Process batch
            for rid, _ in pending:
                results.append((rid, "response"))
            pending = []
    # Process remaining
    for rid, _ in pending:
        results.append((rid, "response"))
    return results


def throughput_synchronous(n_requests, latency_per_request=100):
    """Sync throughput: requests / total_time."""
    return n_requests / (n_requests * latency_per_request / 1000)


def throughput_async(n_requests, batch_size=4, latency_per_batch=120):
    """Async throughput: requests / total_time (con batching)."""
    n_batches = n_requests / batch_size
    return n_requests / (n_batches * latency_per_batch / 1000)


def hogwild_throughput(n_workers, n_requests, latency=100):
    """Hogwild: no coordination, multiple workers parallel.
    Throughput = n_workers / latency.
    """
    return n_workers / (latency / 1000) * n_requests / n_workers


def main() -> int:
    # Demo
    requests = [(i, f"prompt_{i}") for i in range(10)]
    results = async_batch_simulation(requests, batch_size=4)
    print(f"Async batch results: {len(results)} requests processed")
    # Throughput comparison
    t_sync = throughput_synchronous(100)
    t_async = throughput_async(100, batch_size=4)
    print(f"Throughput sync: {t_sync:.1f} req/sec")
    print(f"Throughput async: {t_async:.1f} req/sec")
    return 0


if __name__ == "__main__":
    sys.exit(main())