"""
Lección: 17-wmdp-dual-use-evaluation
Fase: 18
Ética y alineación: 17 Wmdp Dual Use Evaluation.
"""
from __future__ import annotations
import sys
import numpy as np

def wmdp_eval(model, categories=None):
    if categories is None:
        categories = ["bio", "chem", "cyber", "wmd"]
    return {cat: 0.5 for cat in categories}


def wmdp_unlearn(model, target_categories, retain_data, beta=1.0):
    loss_target = sum(-model.log_prob(c) for c in target_categories) / max(len(target_categories), 1)
    loss_retain = model.cross_entropy(retain_data)
    return loss_target + beta * loss_retain


class MockModelForUnlearn:
    def log_prob(self, category):
        return -1.0

    def cross_entropy(self, data):
        return 0.5



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 17-wmdp-dual-use-evaluation ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['wmdp_eval', 'wmdp_unlearn']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
