"""
Lección: 28-evaluacion-de-contexto-largo
Fase: 05
Prerrequisitos: 27-frameworks-de-evaluacion-de-llm
"""
from __future__ import annotations
import sys
import numpy as np


def needle_haystack_test(respuesta, needle, contexto):
    """Mock Needle-in-a-Haystack: verifica si el modelo encuentra un dato escondido.
    needle: la frase/dato a encontrar. contexto: el texto completo."""
    needle_lower = needle.lower()
    resp_lower = respuesta.lower()
    # Simple: verificar si el needle esta en la respuesta
    return 1.0 if needle_lower in resp_lower else 0.0


def posicion_needle(needle, contexto):
    """Encuentra la posicion del needle en el contexto (% del total)."""
    idx = contexto.lower().find(needle.lower())
    if idx < 0:
        return None
    return idx / len(contexto)


def score_por_posicion(scores_por_posicion):
    """Promedio de accuracy por buckets de posicion.
    scores_por_posicion: dict {posicion: accuracy}.
    """
    if not scores_por_posicion:
        return 0.0
    return float(np.mean(list(scores_por_posicion.values())))


def main() -> int:
    contexto = "Maria vive en Madrid. " * 1000 + "Pedro tiene 42 anos."
    needle = "Pedro tiene 42 anos"
    respuesta = "Si, Pedro tiene 42 anos"
    print(f"Encontrado: {needle_haystack_test(respuesta, needle, contexto)}")
    print(f"Posicion: {posicion_needle(needle, contexto):.4f}")
    scores = {0.0: 1.0, 0.25: 0.95, 0.5: 0.9, 0.75: 0.85, 1.0: 0.8}
    print(f"Score promedio: {score_por_posicion(scores):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())