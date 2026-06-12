"""
Lección: 28-alignment-research-ecosystem
Fase: 00
Prerrequisitos: Ninguno
Fuentes: <añadir URLs a papers, RFCs o docs oficiales>
"""
from __future__ import annotations

import sys


def main() -> int:
    """Demo auto-terminal. Imprime un resumen del entorno."""
    print("=== 28-alignment-research-ecosystem ===")
    print(f"Python {sys.version.split()[0]}")
    print(f"Plataforma: {sys.platform}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
