#!/usr/bin/env python3
"""
Generador rápido de lecciones: rellena el esqueleto de scaffold con
contenido pedagogico en espanol.

Uso:
    python3 scripts/generar_contenido_leccion.py \
        --fase 01 --slug 02-vectores-matrices-operaciones \
        --titulo "Vectores, matrices y operaciones" \
        --titulo-original "Vectors, Matrices and Operations" \
        --tipo Construir --lenguajes python \
        --tiempo 45 \
        --problema "..." \
        --concepto "..." \
        --codigo-path scripts/snippets/vectores.py \
        --prompt outputs/prompt-vectores.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def cabecera_lesson(titulo: str, lema: str, tipo: str, lenguajes: list[str],
                    prereq: str, tiempo: int) -> str:
    return f"""# {titulo}

> {lema}

**Tipo:** {tipo}
**Lenguajes:** {", ".join(lenguajes)}
**Prerrequisitos:** {prereq}
**Tiempo estimado:** ~{tiempo} minutos

"""


def escribir_leccion(args: argparse.Namespace) -> None:
    fase_dir = next(RAIZ.glob(f"fases/{args.fase}-*"))
    lec_dir = fase_dir / args.slug
    if not lec_dir.exists():
        sys.exit(f"Primero ejecuta scaffold_leccion.py para {args.slug}")

    docs = lec_dir / "docs" / "es.md"
    tests = lec_dir / "code" / "tests" / "test_main.py"
    main_py = lec_dir / "code" / "main.py"
    outputs = lec_dir / "outputs"

    # docs/es.md se sobreescribe con la plantilla pedagogica
    contenido = args.docs
    docs.write_text(contenido, encoding="utf-8")

    # main.py
    if args.main_py:
        main_py.write_text(args.main_py, encoding="utf-8")

    # tests
    if args.tests_py:
        tests.parent.mkdir(parents=True, exist_ok=True)
        tests.write_text(args.tests_py, encoding="utf-8")

    # output prompt
    if args.prompt_md:
        outputs.mkdir(exist_ok=True)
        (outputs / f"prompt-{args.slug.split('-', 1)[1]}.md").write_text(
            args.prompt_md, encoding="utf-8"
        )

    # quiz
    if args.quiz_json:
        (lec_dir / "quiz.json").write_text(args.quiz_json, encoding="utf-8")

    print(f"  OK: {lec_dir.relative_to(RAIZ)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fase", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--main-py")
    parser.add_argument("--tests-py")
    parser.add_argument("--prompt-md")
    parser.add_argument("--quiz-json")
    args = parser.parse_args()
    escribir_leccion(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
