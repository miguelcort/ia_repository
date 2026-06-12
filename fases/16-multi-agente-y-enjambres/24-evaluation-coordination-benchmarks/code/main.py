"""
Lección: 24-evaluation-coordination-benchmarks
Fase: 16
Evaluation coordination benchmarks:
MGSM, HumanEval-Multi, MLE-bench,
ChatBot Arena Multi-turn, multi-agent
specific metrics.
"""
from __future__ import annotations


COORDINATION_BENCHMARKS = {
    "MGSM": {
        "name": "Multi-Grade School Math",
        "description": "Multi-step math reasoning",
        "metric": "accuracy",
        "year": 2023,
    },
    "HumanEval-Multi": {
        "name": "HumanEval Multi",
        "description": "Multi-file code generation",
        "metric": "pass@k",
        "year": 2024,
    },
    "MLE-bench": {
        "name": "MLE-bench",
        "description": "ML engineering tasks",
        "metric": "score",
        "year": 2024,
    },
    "ChatBot-Arena-MT": {
        "name": "ChatBot Arena Multi-turn",
        "description": "Multi-turn conversation quality",
        "metric": "elo",
        "year": 2024,
    },
    "SWE-bench-Multi": {
        "name": "SWE-bench Multi",
        "description": "Multi-repo software engineering",
        "metric": "resolve_rate",
        "year": 2024,
    },
}


def list_benchmarks():
    return list(COORDINATION_BENCHMARKS.keys())


def get_benchmark(key):
    return COORDINATION_BENCHMARKS.get(key)


def normalize_score(value, benchmark_key, scale=100):
    """Normalize score to 0-1 range."""
    bench = COORDINATION_BENCHMARKS.get(benchmark_key)
    if not bench:
        return None
    metric = bench["metric"]
    if metric in ("accuracy", "pass@k", "score", "resolve_rate"):
        return min(1.0, max(0.0, value / scale))
    if metric == "elo":
        return min(1.0, max(0.0, (value - 800) / (1600 - 800)))
    return None


def average_coordination_score(scores):
    """Average normalized scores across benchmarks."""
    if not scores:
        return 0.0
    values = [v for v in scores.values() if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> int:
    print(f"Benchmarks: {list_benchmarks()}")
    print(f"Normalize 80: {normalize_score(80, 'MGSM')}")
    print(f"Avg: {average_coordination_score({'MGSM': 0.8, 'HumanEval-Multi': 0.6})}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())