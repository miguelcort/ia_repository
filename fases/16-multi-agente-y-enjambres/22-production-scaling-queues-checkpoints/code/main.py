"""
Lección: 22-production-scaling-queues-checkpoints
Fase: 16
Production scaling: queues for
backpressure, horizontal scaling,
checkpoint state, graceful shutdown,
multi-agent at scale.
"""
from __future__ import annotations
import time
import uuid


class Queue:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.items = []
        self.dropped = 0

    def put(self, item, timeout=None):
        if len(self.items) >= self.max_size:
            self.dropped += 1
            return False
        self.items.append(item)
        return True

    def get(self):
        if not self.items:
            return None
        return self.items.pop(0)

    def size(self):
        return len(self.items)

    def full(self):
        return len(self.items) >= self.max_size


class Worker:
    def __init__(self, agent_id, queue):
        self.agent_id = agent_id
        self.queue = queue
        self.processed = 0
        self.running = False

    def step(self):
        if not self.running:
            return None
        item = self.queue.get()
        if item is None:
            return None
        self.processed += 1
        return item

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


def horizontal_scale(workers, queue, max_workers=10, scale_up_threshold=5, scale_down_threshold=1):
    """Add or remove workers based on queue depth."""
    while len(workers) < max_workers and queue.size() > scale_up_threshold:
        workers.append(Worker(str(uuid.uuid4()), queue))
    while len(workers) > 0 and queue.size() < scale_down_threshold:
        w = workers.pop()
        w.stop()


def graceful_shutdown(workers, queue, drain_timeout=5.0):
    """Stop accepting new work, drain queue, then stop workers."""
    for w in workers:
        w.stop()
    start = time.time()
    while queue.size() > 0 and time.time() - start < drain_timeout:
        time.sleep(0.01)


class CheckpointStore:
    def __init__(self):
        self.checkpoints = {}

    def save(self, agent_id, state):
        self.checkpoints[agent_id] = {"state": dict(state), "ts": time.time()}

    def load(self, agent_id):
        return self.checkpoints.get(agent_id)


def main() -> int:
    q = Queue(max_size=10)
    for i in range(3):
        q.put({"task": i})
    w = Worker("a1", q)
    w.start()
    print(w.step())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())