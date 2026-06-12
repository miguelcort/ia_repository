"""
Lección: 21-metr-external-evaluation
Fase: 15
METR external evaluation: independent
measurement of AI capabilities,
task suite, time horizon, agent
benchmarks, HCAST, RE-Bench.
"""
from __future__ import annotations


METR_BENCHMARKS = {
    "HCAST": {
        "name": "Human-Calibrated Agent Solver Tasks",
        "description": "Tasks with human time estimates to calibrate AI capability",
        "domain": "general",
        "metric": "time_horizon",
        "year": 2024,
    },
    "RE-Bench": {
        "name": "Research Engineering Benchmark",
        "description": "ML research engineering tasks",
        "domain": "research",
        "metric": "score",
        "year": 2024,
    },
    "SWE-bench": {
        "name": "Software Engineering Benchmark",
        "description": "Real GitHub issues resolution",
        "domain": "coding",
        "metric": "resolve_rate",
        "year": 2023,
    },
    "GAIA": {
        "name": "GAIA: General AI Assistants",
        "description": "Multi-modal agent tasks",
        "domain": "general",
        "metric": "accuracy",
        "year": 2023,
    },
}


def list_benchmarks():
    return list(METR_BENCHMARKS.keys())


def get_benchmark(key):
    return METR_BENCHMARKS.get(key)


def time_horizon_score(tasks, model_times):
    """Return median time horizon given task durations and model times."""
    horizons = []
    for task, duration in tasks.items():
        if task in model_times:
            horizon = duration / max(model_times[task], 0.01)
            horizons.append(horizon)
    if not horizons:
        return 0.0
    horizons.sort()
    n = len(horizons)
    if n % 2 == 0:
        return (horizons[n//2 - 1] + horizons[n//2]) / 2
    return horizons[n//2]


def evaluate_model(model_results, benchmark_key):
    """Return metric value from model results on a benchmark."""
    bench = METR_BENCHMARKS.get(benchmark_key)
    if not bench:
        raise ValueError(f"unknown benchmark: {benchmark_key}")
    return {
        "benchmark": benchmark_key,
        "metric": bench["metric"],
        "value": model_results.get(benchmark_key, 0.0),
    }


def main() -> int:
    print(f"Benchmarks: {list_benchmarks()}")
    tasks = {"task1": 60, "task2": 300, "task3": 600}
    model_times = {"task1": 10, "task2": 60, "task3": 120}
    print(f"Time horizon: {time_horizon_score(tasks, model_times):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())