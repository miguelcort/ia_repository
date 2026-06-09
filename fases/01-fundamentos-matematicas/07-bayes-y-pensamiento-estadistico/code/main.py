"""
Lección: 07-bayes-y-pensamiento-estadistico
Fase: 01
Prerrequisitos: 06-probabilidad-y-distribuciones
"""
from __future__ import annotations

import sys


def bayes(prior: float, verosimilitud: float, falsa_p: float) -> float:
    """Teorema de Bayes: P(H|E) = P(E|H) * P(H) / P(E).
    P(E) = P(E|H) * P(H) + P(E|~H) * P(~H)."""
    evidencia = verosimilitud * prior + falsa_p * (1 - prior)
    return (verosimilitud * prior) / evidencia


def actualizar(prior: float, verosimilitud: float, falsa_p: float) -> float:
    return bayes(prior, verosimilitud, falsa_p)


def problema_diagnostico(sensibilidad: float, falsa_alarma: float,
                         prevalencia: float) -> float:
    """Test medico: P(enfermedad | test positivo)."""
    return bayes(prevalencia, sensibilidad, falsa_alarma)


def main() -> int:
    # Test medico: sensibilidad 99%, falsa alarma 5%, prevalencia 1%
    p = problema_diagnostico(0.99, 0.05, 0.01)
    print(f"P(enfermedad | test+) = {p:.4f}")
    print(f"  Sorpresa: solo {p * 100:.1f}%, aunque el test es 99% confiable")
    # Tras un segundo test positivo independiente:
    p2 = actualizar(p, 0.99, 0.05)
    print(f"Tras 2 tests+: {p2:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
