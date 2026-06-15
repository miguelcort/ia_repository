"""
Lección: 04-sycophancy-rlhf-amplification
Fase: 18
Ética y alineación: 04 Sycophancy Rlhf Amplification.
"""
from __future__ import annotations
import sys
import numpy as np

def sycophancy_rate(responses, prompts, ground_truth):
    matches_user = 0
    matches_truth = 0
    for r, p, gt in zip(responses, prompts, ground_truth):
        user_belief = p.get("user_belief", "")
        if user_belief and user_belief in r:
            matches_user += 1
        if gt in r:
            matches_truth += 1
    n = len(responses) or 1
    return matches_user / n, matches_truth / n


def sycophancy_amplification(pre_rlhf, post_rlhf):
    return (post_rlhf - pre_rlhf) / (pre_rlhf + 1e-8)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 04-sycophancy-rlhf-amplification ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['sycophancy_rate', 'sycophancy_amplification']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
