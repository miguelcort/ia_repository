"""
Lección: 24-chaos-engineering-llm
Fase: 17
Chaos engineering for LLM: inject
failures, latency, malformed responses,
test recovery, game days, blast radius.
"""
from __future__ import annotations
import random


class ChaosExperiment:
    def __init__(self, name, hypothesis):
        self.name = name
        self.hypothesis = hypothesis
        self.injectors = []
        self.results = []

    def add_injector(self, injector):
        self.injectors.append(injector)

    def run(self, target_fn, n_trials=10):
        successes = 0
        for _ in range(n_trials):
            try:
                for inj in self.injectors:
                    inj()
                target_fn()
                successes += 1
            except Exception as e:
                self.results.append({"ok": False, "error": str(e)})
        return successes / n_trials


def inject_latency(ms):
    """Add latency to next call."""
    import time
    delay = ms / 1000
    original_sleep = time.sleep
    time.sleep = lambda x: original_sleep(x + delay)


def inject_error():
    """Raise exception on next call."""
    raise RuntimeError("chaos: injected error")


def inject_malformed_response():
    """Return malformed response."""
    return {"unexpected_field": "value"}


class ChaosMonkey:
    def __init__(self, seed=None):
        self.experiments = []
        if seed is not None:
            random.seed(seed)

    def register(self, experiment):
        self.experiments.append(experiment)

    def run_all(self, target_fn):
        results = []
        for exp in self.experiments:
            rate = exp.run(target_fn)
            results.append({"name": exp.name, "success_rate": rate})
        return results


def main() -> int:
    monkey = ChaosMonkey(seed=42)
    exp = ChaosExperiment("error_injection", "system handles errors gracefully")
    exp.add_injector(inject_error)
    monkey.register(exp)
    print(monkey.run_all(lambda: None))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())