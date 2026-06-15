"""
Lección: 71-classical-metrics
Fase: 19
Capstone de ingeniería AI: 71 Classical Metrics.
"""
from __future__ import annotations
import sys

def exact_match(pred, target):
    return int(pred.strip().lower() == target.strip().lower())


def f1_score(pred, target):
    pred_tokens = pred.lower().split()
    target_tokens = target.lower().split()
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(target_tokens)
    return 2 * prec * rec / (prec + rec)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
