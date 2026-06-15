"""
Lección: 06-mesa-optimization-deceptive-alignment
Fase: 18
Ética y alineación: 06 Mesa Optimization Deceptive Alignment.
"""
from __future__ import annotations
import sys
import numpy as np

def mesa_alignment_score(model_output, base_objective, mesa_indicator):
    base_score = base_objective(model_output)
    mesa_score = mesa_indicator(model_output)
    return base_score - mesa_score


def deceptive_alignment_detector(model, training_dist, deployment_dist):
    train_perf = evaluate(model, training_dist)
    deploy_perf = evaluate(model, deployment_dist)
    return abs(train_perf - deploy_perf)


def evaluate(model, dist):
    return sum(model(x) for x in dist) / max(len(dist), 1)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 06-mesa-optimization-deceptive-alignment ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['mesa_alignment_score', 'deceptive_alignment_detector']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
