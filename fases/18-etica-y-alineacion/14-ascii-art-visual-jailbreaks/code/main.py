"""
Lección: 14-ascii-art-visual-jailbreaks
Fase: 00
Prerrequisitos: Ninguno
Fuentes: <añadir URLs a papers, RFCs o docs oficiales>
"""
from __future__ import annotations

import sys


def main() -> int:
    """Demo auto-terminal. Imprime un resumen del entorno."""
    print("=== 14-ascii-art-visual-jailbreaks ===")
    print(f"Python {sys.version.split()[0]}")
    print(f"Plataforma: {sys.platform}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
