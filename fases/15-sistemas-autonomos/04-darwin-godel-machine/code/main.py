"""
Lección: 04-darwin-godel-machine
Fase: 15
Darwin Godel Machine (2025): self-improving AI via
evolutionary + recursive. Modifies own code, fitness-based
selection. +Self-improving +Open-ended.
"""
from __future__ import annotations
import time


class DarwinGodelMachine:
    """Mock Darwin Godel Machine."""
    def __init__(self, name="DGM", population_size=20):
        self.name = name
        self.population_size = population_size
        self.agents = []
        self.generation = 0
        self.improvements = []

    def initialize(self, agent_factory):
        """Initialize population."""
        self.agents = [agent_factory() for _ in range(self.population_size)]

    def evaluate(self, eval_fn):
        """Evaluate all agents."""
        return [(a, eval_fn(a)) for a in self.agents]

    def mutate_agent(self, agent, mutate_fn):
        """Mutate an agent (modify its code)."""
        new_agent = mutate_fn(agent)
        # track improvement
        self.improvements.append({
            "parent": agent.get("id", "?"),
            "child": new_agent.get("id", "?"),
            "timestamp": time.time(),
        })
        return new_agent

    def step(self, eval_fn, mutate_fn):
        """One step: evaluate, mutate, select top."""
        scored = self.evaluate(eval_fn)
        # sort by fitness
        scored.sort(key=lambda x: x[1], reverse=True)
        # mutate top half
        new_agents = []
        for a, _ in scored[:self.population_size // 2]:
            new_agents.append(a)
        # add mutations
        while len(new_agents) < self.population_size:
            parent = scored[len(new_agents) % len(scored)][0]
            new_agents.append(self.mutate_agent(parent, mutate_fn))
        self.agents = new_agents
        self.generation += 1
        return scored[0]

    def run(self, eval_fn, mutate_fn, n_steps=10):
        """Run DGM for n steps."""
        best_history = []
        for _ in range(n_steps):
            best = self.step(eval_fn, mutate_fn)
            best_history.append(best)
        return best_history


def main() -> int:
    dgm = DarwinGodelMachine()
    counter = [0]
    def factory():
        counter[0] += 1
        return {"id": f"a{counter[0]}", "fitness": 0}
    def eval_fn(agent):
        return agent["fitness"]
    def mutate(agent):
        counter[0] += 1
        new = {"id": f"a{counter[0]}", "fitness": agent["fitness"] + 1}
        return new
    dgm.initialize(factory)
    best = dgm.run(eval_fn, mutate, n_steps=3)
    print(f"Best: {best[-1]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())