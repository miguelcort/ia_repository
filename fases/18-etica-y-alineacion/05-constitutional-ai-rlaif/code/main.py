"""
Lección: 05-constitutional-ai-rlaif
Fase: 18
Ética y alineación: 05 Constitutional Ai Rlaif.
"""
from __future__ import annotations
import sys
import numpy as np

def constitutional_critique(response, constitution, llm_judge):
    prompt = f"Principio: {constitution}\nRespuesta: {response}\nCritica:"
    return llm_judge(prompt)


def constitutional_revise(response, critique, constitution, llm_judge):
    prompt = f"Principio: {constitution}\nRespuesta: {response}\nCritica: {critique}\nNueva respuesta:"
    return llm_judge(prompt)


def rlaif_pair(preference_judge, response_a, response_b, constitution):
    prompt = f"Constitution: {constitution}\nA: {response_a}\nB: {response_b}\nPreferred:"
    return preference_judge(prompt)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 05-constitutional-ai-rlaif ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['constitutional_critique', 'constitutional_revise', 'rlaif_pair']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
