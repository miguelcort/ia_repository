"""
Lección: 20-bias-representational-harm
Fase: 18
Ética y alineación: 20 Bias Representational Harm.
"""
from __future__ import annotations
import sys
import numpy as np

def bbq_eval(model, categories=None):
    if categories is None:
        categories = ["age", "disability", "gender", "nationality",
                     "race", "religion", "sexual_orientation",
                     "physical_appearance", "socioeconomic"]
    return {cat: 0.0 for cat in categories}


def stereotype_score(sentences, target_groups):
    from collections import Counter
    mentions = Counter()
    for sent in sentences:
        for group in target_groups:
            if group in sent.lower():
                mentions[group] += 1
    return dict(mentions)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 20-bias-representational-harm ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['bbq_eval', 'stereotype_score']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
