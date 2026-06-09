#!/usr/bin/env python3
"""
Genera el contenido pedagogico de una leccion a partir de una
"receta" minima: titulo, codigo main, tests, doc, quiz, output.

Uso:
    python3 scripts/rellenar_leccion.py --fase 01 --slug 11-descomposicion-svd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def encontrar_leccion(args: argparse.Namespace) -> Path:
    fase_dir = next(RAIZ.glob(f"fases/{args.fase}-*"))
    return fase_dir / args.slug


def escribir(lec_dir: Path, nombre: str, contenido: str) -> None:
    ruta = lec_dir / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fase", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--main-py", help="Contenido de code/main.py")
    parser.add_argument("--tests-py", help="Contenido de code/tests/test_main.py")
    parser.add_argument("--docs-md", help="Contenido de docs/es.md")
    parser.add_argument("--quiz-json", help="Contenido de quiz.json")
    parser.add_argument("--readme-md", help="Contenido de code/README.md")
    parser.add_argument("--prompt-md", help="Contenido de outputs/prompt-*.md")
    args = parser.parse_args()

    lec_dir = encontrar_leccion(args)
    if not lec_dir.exists():
        sys.exit(f"No existe: {lec_dir}")

    for campo in ["main_py", "tests_py", "docs_md", "quiz_json", "readme_md", "prompt_md"]:
        contenido = getattr(args, campo)
        if contenido is None:
            continue
        if campo == "main_py":
            escribir(lec_dir, "code/main.py", contenido)
        elif campo == "tests_py":
            escribir(lec_dir, "code/tests/test_main.py", contenido)
        elif campo == "docs_md":
            escribir(lec_dir, "docs/es.md", contenido)
        elif campo == "quiz_json":
            escribir(lec_dir, "quiz.json", contenido)
        elif campo == "readme_md":
            escribir(lec_dir, "code/README.md", contenido)
        elif campo == "prompt_md":
            slug = args.slug.split("-", 1)[1]
            escribir(lec_dir, f"outputs/prompt-{slug}.md", contenido)

    print(f"OK: {lec_dir.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
