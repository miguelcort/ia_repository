"""
Lección: 16-evaluacion-de-modelos-generativos
Fase: 08
Benchmarks y estudios humanos para modelos generativos.
"""
from __future__ import annotations
import sys
import numpy as np
from math import erf, sqrt


def pair_preference_pvalue(wins_a, wins_b, ties=0, alpha=0.05):
    """Evaluar si modelo A es significativamente mejor que B.
    Test: z-test aproximado (correction de continuidad).
    Returns: p_value.
    """
    n = wins_a + wins_b
    p_hat = wins_a / n
    z = (p_hat - 0.5) / np.sqrt(0.25 / n)
    return 0.5 * (1 - erf(z / np.sqrt(2)))


def elo_rating(rating_a, rating_b, score_a, k=32):
    """Elo rating update.
    score_a: 1 = A wins, 0 = B wins, 0.5 = tie.
    """
    exp_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def mos_score(ratings, scale_max=5):
    """Mean Opinion Score: promedio de ratings humanos.
    ratings: array de scores. scale_max: maximo.
    """
    return float(np.mean(ratings))


def compute_confidence_interval(scores, confidence=0.95):
    """Bootstrap confidence interval.
    """
    rng = np.random.default_rng(0)
    n = len(scores)
    n_boot = 1000
    means = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def benchmark_summary():
    """Benchmarks para modelos generativos."""
    return {
        "FID": "Frechet Inception Distance, distribucional",
        "CLIP score": "Text-image alignment",
        "ImageReward": "Reward model para text-to-image",
        "HPSv2": "Human Preference Score v2",
        "PickScore": "PickScore (CLIP-based preference)",
        "HEIM": "Holistic Evaluation of Image Models (12 axes)",
        "T2I-CompBench": "Compositionality benchmark",
        "GenEval": "Generation evaluation (object, count, etc)",
    }


def main() -> int:
    print("=== Benchmarks ===")
    for k, v in benchmark_summary().items():
        print(f"  {k:18s} {v}")
    # Demo Elo
    rating_a, rating_b = 1500, 1500
    rating_a, rating_b = elo_rating(rating_a, rating_b, score_a=1.0)
    print(f"\nDespues de A gana: A={rating_a:.0f}, B={rating_b:.0f}")
    # MOS
    scores = np.random.default_rng(0).uniform(3, 5, size=50)
    print(f"MOS: {mos_score(scores):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())