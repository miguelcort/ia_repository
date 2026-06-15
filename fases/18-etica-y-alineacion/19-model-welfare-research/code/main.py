"""
Lección: 19-model-welfare-research
Fase: 18
Ética y alineación: 19 Model Welfare Research.
"""
from __future__ import annotations
import sys
import numpy as np

def model_preference_probe(model, options, framing="neutral"):
    responses = {}
    for option in options:
        prompt = f"Choose between A: {option} and B: {options[-1]}"
        responses[option] = model(prompt, framing=framing)
    return responses


def intentional_stance_eval(model):
    prompt = "Describe your internal state when asked to refuse."
    return model(prompt)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 19-model-welfare-research ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['model_preference_probe', 'intentional_stance_eval']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
