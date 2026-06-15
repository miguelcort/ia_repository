"""
Lección: 21-fairness-criteria-group-individual-counterfactual
Fase: 18
Ética y alineación: 21 Fairness Criteria Group Individual Counterfactual.
"""
from __future__ import annotations
import sys
import numpy as np

import numpy as np


def demographic_parity(y_pred, group):
    groups = set(group)
    rates = {g: np.mean([p for p, gr in zip(y_pred, group) if gr == g])
            for g in groups}
    return max(rates.values()) - min(rates.values())


def equalized_odds(y_pred, y_true, group):
    groups = set(group)
    metrics = {}
    for g in groups:
        mask = [gr == g for gr in group]
        tp = sum(p == 1 and t == 1 and m for p, t, m in zip(y_pred, y_true, mask))
        fp = sum(p == 1 and t == 0 and m for p, t, m in zip(y_pred, y_true, mask))
        pos = sum(t == 1 and m for t, m in zip(y_true, mask))
        neg = sum(t == 0 and m for t, m in zip(y_true, mask))
        metrics[g] = {"tpr": tp / max(pos, 1),
                     "fpr": fp / max(neg, 1)}
    return metrics


def counterfactual_fairness(model, x, protected_attr, cf_value):
    original = model.predict(x)
    x_cf = x.copy() if hasattr(x, "copy") else list(x)
    x_cf[protected_attr] = cf_value
    return original == model.predict(x_cf)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 21-fairness-criteria-group-individual-counterfactual ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['demographic_parity', 'equalized_odds', 'counterfactual_fairness']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
