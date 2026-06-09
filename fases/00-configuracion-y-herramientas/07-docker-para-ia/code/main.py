"""
Lección: 07-docker-para-ia
Fase: 00
Prerrequisitos: 01-entorno-desarrollo, 06-entornos-python
Fuentes:
- Docker docs: https://docs.docker.com/reference/dockerfile/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo de la lección 07-docker-para-ia")
    parser.add_argument("--version", action="store_true", help="Imprime la versión y sale")
    args = parser.parse_args(argv)

    if args.version:
        info = {
            "app": "leccion-07-docker-para-ia",
            "version": "0.1.0",
            "python": sys.version.split()[0],
            "sistema": platform.platform(),
            "docker": bool(os.environ.get("DOCKER_CONTAINER")),
        }
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print("Leccion 07: Docker para IA")
    print("  Construir: docker build -t mi-app .")
    print("  Ejecutar:  docker run --rm mi-app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
