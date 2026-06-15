"""
Lección: 72-code-exec-metric
Fase: 19
Capstone de ingeniería AI: 72 Code Exec Metric.
"""
from __future__ import annotations
import sys

from math import comb


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def execute_code_test(code, test_cases, timeout=10):
    results = []
    for test in test_cases:
        try:
            ns = {}
            exec(code, ns)
            result = ns["solve"](*test["args"])
            results.append(result == test["expected"])
        except Exception:
            results.append(False)
    return sum(results) / len(test_cases)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
