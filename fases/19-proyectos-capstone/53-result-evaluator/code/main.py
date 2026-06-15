"""
Lección: 53-result-evaluator
Fase: 19
Capstone de ingeniería AI: 53 Result Evaluator.
"""
from __future__ import annotations
import sys

import numpy as np
from scipy import stats


def compare_models(metric_a, metric_b, alpha=0.05):
    t_stat, p_value = stats.ttest_rel(metric_a, metric_b)
    d = (np.mean(metric_a) - np.mean(metric_b)) / np.std(metric_b)
    diffs = np.array(metric_a) - np.array(metric_b)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {"p_value": p_value, "effect_size": d,
            "ci_95": (ci_low, ci_high),
            "significant": p_value < alpha}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
