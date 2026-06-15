"""
Lección: 74-leaderboard-aggregation
Fase: 19
Capstone de ingeniería AI: 74 Leaderboard Aggregation.
"""
from __future__ import annotations
import sys

def average_score(model_results):
    return sum(model_results.values()) / len(model_results)


def weighted_score(model_results, weights):
    return sum(model_results[k] * w for k, w in weights.items())


def elo_update(rating_a, rating_b, score_a, k=32):
    exp_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
