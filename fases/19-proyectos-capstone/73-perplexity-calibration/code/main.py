"""
Lección: 73-perplexity-calibration
Fase: 19
Capstone de ingeniería AI: 73 Perplexity Calibration.
"""
from __future__ import annotations
import sys

import numpy as np


def perplexity(losses):
    return np.exp(np.mean(losses))


def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (labels[mask] == 1).mean()
        conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return ece



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
