"""
Lección: 03-direct-preference-optimization-family
Fase: 18
Ética y alineación: 03 Direct Preference Optimization Family.
"""
from __future__ import annotations
import sys
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_prob(logits, target):
    log_softmax = logits - np.log(np.exp(logits).sum(-1) + 1e-8)
    return log_softmax[target]


def dpo_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1):
    diff = beta * ((policy_chosen - ref_chosen)
                  - (policy_rejected - ref_rejected))
    return -np.log(sigmoid(diff) + 1e-8).mean()


def ipo_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1, tau=0.1):
    diff = (policy_chosen - ref_chosen
           - (policy_rejected - ref_rejected)) / (2 * beta)
    return (diff ** 2).mean() / 2 + tau


def kto_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1, desirable=True):
    kl_chosen = beta * (policy_chosen - ref_chosen)
    kl_rejected = beta * (policy_rejected - ref_rejected)
    if desirable:
        return -np.log(sigmoid(kl_chosen) + 1e-8).mean()
    return -np.log(1 - sigmoid(kl_rejected) + 1e-8).mean()


def orpo_loss(policy_chosen, policy_rejected,
            sft_chosen, sft_rejected, beta=0.1):
    log_odds_chosen = (policy_chosen - sft_chosen)
    log_odds_rejected = (policy_rejected - sft_rejected)
    return -np.log(sigmoid(beta * (log_odds_chosen
                                  - log_odds_rejected))
                  + 1e-8).mean()


def simpo_loss(policy_chosen, policy_rejected, beta=2.0,
            gamma=1.0):
    return -np.log(sigmoid(beta * policy_chosen - gamma
                          - (beta * policy_rejected - gamma))
                  + 1e-8).mean()



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 03-direct-preference-optimization-family ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['sigmoid', 'log_prob', 'dpo_loss', 'ipo_loss', 'kto_loss', 'orpo_loss', 'simpo_loss']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
