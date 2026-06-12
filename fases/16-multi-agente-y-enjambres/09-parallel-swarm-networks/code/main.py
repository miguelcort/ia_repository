"""
Lección: 09-parallel-swarm-networks
Fase: 16
Parallel swarm networks: many agents
running in parallel, fan-out/fan-in,
result aggregation, throughput
maximization.
"""
from __future__ import annotations
import time


def fan_out(agents, task):
    """Send the same task to all agents in parallel."""
    results = []
    for a in agents:
        results.append({"agent": a, "result": a(task)})
    return results


def fan_in(results, reducer="concat"):
    """Aggregate parallel results."""
    if reducer == "concat":
        return [r["result"] for r in results]
    if reducer == "first":
        return results[0]["result"] if results else None
    if reducer == "majority":
        counts = {}
        for r in results:
            v = r["result"]
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get) if counts else None
    return results


class Swarm:
    def __init__(self, agents, reducer="concat"):
        self.agents = agents
        self.reducer = reducer
        self.history = []

    def run(self, task):
        results = fan_out(self.agents, task)
        final = fan_in(results, self.reducer)
        self.history.append({"task": task, "results": results, "final": final})
        return final

    def run_many(self, tasks):
        return [self.run(t) for t in tasks]


def main() -> int:
    agents = [
        lambda t: f"a1:{t}",
        lambda t: f"a2:{t}",
        lambda t: f"a3:{t}",
    ]
    s = Swarm(agents, reducer="concat")
    print(s.run("hello"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())