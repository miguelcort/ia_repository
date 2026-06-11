"""
Lección: 28-orchestration-patterns
Fase: 14
Orchestration patterns: manager-worker, pipeline, scatter-gather,
event-driven. Multi-agent coordination.
+Production +Scalable.
"""
from __future__ import annotations
import time
from collections import defaultdict


class Orchestrator:
    """Mock orchestrator."""
    def __init__(self, name="orchestrator"):
        self.name = name
        self.workers = {}
        self.history = []

    def register_worker(self, name, worker_fn):
        self.workers[name] = worker_fn

    def manager_worker(self, task, n_workers=3):
        """Manager-worker: manager decomposes, workers execute."""
        # decompose
        subtasks = [f"subtask_{i}_{task}" for i in range(n_workers)]
        results = []
        for sub in subtasks:
            # round-robin assignment
            worker_name = list(self.workers.keys())[hash(sub) % len(self.workers)]
            result = self.workers[worker_name](sub)
            results.append(result)
        return results

    def pipeline(self, stages, input_data):
        """Pipeline: stage1 -> stage2 -> stage3."""
        result = input_data
        for stage in stages:
            worker_name = stage
            if worker_name in self.workers:
                result = self.workers[worker_name](result)
        return result

    def scatter_gather(self, task, n_workers=3):
        """Scatter-gather: scatter task, gather results."""
        # scatter
        results = []
        worker_names = list(self.workers.keys())
        for i in range(n_workers):
            w = worker_names[i % len(worker_names)]
            r = self.workers[w](f"{task}_{i}")
            results.append(r)
        # gather
        return {"task": task, "results": results}


def main() -> int:
    orch = Orchestrator()
    orch.register_worker("worker1", lambda x: f"w1({x})")
    orch.register_worker("worker2", lambda x: f"w2({x})")
    orch.register_worker("worker3", lambda x: f"w3({x})")
    # Manager-worker
    results = orch.manager_worker("task_a", n_workers=5)
    print(f"Manager-worker: {len(results)} results")
    # Pipeline
    pipe = orch.pipeline(["worker1", "worker2"], "data")
    print(f"Pipeline: {pipe}")
    # Scatter-gather
    sg = orch.scatter_gather("task_b", n_workers=3)
    print(f"Scatter-gather: {sg['results']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())