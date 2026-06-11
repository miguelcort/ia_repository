"""
Lección: 30-eval-driven-agent-development
Fase: 14
Eval-driven development: evals, regression, benchmarks, A/B testing.
CI/CD for agents. Continuous improvement.
+Production +Reliable.
"""
from __future__ import annotations
import time
import json


class EvalSuite:
    """Mock eval suite."""
    def __init__(self, name):
        self.name = name
        self.cases = []
        self.history = []

    def add_case(self, case):
        self.cases.append(case)

    def run(self, agent_fn, verbose=False):
        """Run all eval cases."""
        passed = 0
        results = []
        for case in self.cases:
            t0 = time.time()
            try:
                actual = agent_fn(case["input"])
                ok = self._matches(actual, case["expected"])
            except Exception as e:
                actual = f"error: {e}"
                ok = False
            elapsed = time.time() - t0
            r = {
                "name": case["name"],
                "input": case["input"],
                "expected": case["expected"],
                "actual": actual,
                "passed": ok,
                "latency_ms": elapsed * 1000,
            }
            results.append(r)
            if ok:
                passed += 1
            if verbose:
                print(f"  {case['name']}: {'PASS' if ok else 'FAIL'} ({elapsed * 1000:.0f}ms)")
        score = passed / len(self.cases) if self.cases else 0
        summary = {
            "name": self.name,
            "score": score,
            "passed": passed,
            "total": len(self.cases),
            "results": results,
        }
        self.history.append(summary)
        return summary

    def _matches(self, actual, expected):
        if isinstance(expected, str):
            return expected.lower() in str(actual).lower()
        return actual == expected

    def detect_regression(self, threshold=0.05):
        """Detect if latest run has regression vs prior."""
        if len(self.history) < 2:
            return None
        latest = self.history[-1]["score"]
        prior = self.history[-2]["score"]
        return latest < prior - threshold


def main() -> int:
    suite = EvalSuite("my_agent_suite")
    suite.add_case({"name": "greeting", "input": "Hello", "expected": "hi"})
    suite.add_case({"name": "weather", "input": "weather NYC", "expected": "weather"})
    def agent_fn(x):
        return "hi there!" if "Hello" in x else "weather info"
    summary = suite.run(agent_fn, verbose=True)
    print(f"Score: {summary['score']:.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())