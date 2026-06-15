"""
Lección: 11-scalable-oversight-weak-to-strong
Fase: 18
Ética y alineación: 11 Scalable Oversight Weak To Strong.
"""
from __future__ import annotations
import sys
import numpy as np

def weak_to_strong_loss(strong_outputs, weak_labels, strong_gt,
                      beta=0.5):
    aux_loss = cross_entropy(strong_outputs, weak_labels)
    ce_loss = cross_entropy(strong_outputs, strong_gt)
    return aux_loss + beta * ce_loss


def cross_entropy(logits, target):
    import numpy as np
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return -np.log(p[target] + 1e-8).mean()


def debate_protocol(judge_model, two_models, question):
    a = two_models[0](question)
    b = two_models[1](question)
    judge_input = f"Q: {question}\nA: {a}\nB: {b}"
    return judge_model(judge_input)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 11-scalable-oversight-weak-to-strong ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['weak_to_strong_loss', 'debate_protocol']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
