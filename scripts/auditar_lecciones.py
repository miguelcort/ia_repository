#!/usr/bin/env python3
"""
Auditoría estructural del repositorio ia_repository.

Verifica que cada lección tenga la estructura mínima esperada:
- docs/es.md
- code/main.<lang>
- code/tests/test_main.<lang> (con al menos 5 pruebas)
- (opcional) outputs/<artefacto>.md
- (opcional) notebooks/, ejercicios/, referencias/

Uso:
    python3 scripts/auditar_lecciones.py
    python3 scripts/auditar_lecciones.py --fase 02
    python3 scripts/auditar_lecciones.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

RAIZ = Path(__file__).resolve().parent.parent
FASES = RAIZ / "fases"

MIN_PRUEBAS = 5

LECCION_ID = re.compile(r"^(\d{2})-")
FASE_ID = re.compile(r"^(\d{2})-")

EXTENSIONES_CODIGO = {".py", ".ts", ".tsx", ".js", ".rs", ".jl"}


def lecciones_de_fase(fase_dir: Path) -> list[Path]:
    """Devuelve las carpetas de lección dentro de una fase."""
    return sorted(
        p for p in fase_dir.iterdir()
        if p.is_dir() and LECCION_ID.match(p.name)
    )


def archivos_codigo(code_dir: Path) -> list[Path]:
    if not code_dir.exists():
        return []
    return [
        p for p in code_dir.iterdir()
        if p.is_file() and p.suffix in EXTENSIONES_CODIGO
    ]


def contar_pruebas(test_dir: Path) -> int:
    """Cuenta las pruebas (def test_ o fn test_) en un directorio de tests."""
    if not test_dir.exists():
        return 0
    count = 0
    for p in test_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".py", ".rs", ".jl"}:
            try:
                texto = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if p.suffix == ".py":
                count += len(re.findall(r"^\s*def\s+test_\w+", texto, flags=re.M))
            elif p.suffix == ".rs":
                count += len(re.findall(r"#\[test\]", texto))
            elif p.suffix == ".jl":
                count += len(re.findall(r"@test\s+\w+\s*=", texto))
    return count


def auditar_leccion(leccion: Path) -> dict:
    """Devuelve un dict con los hallazgos para una lección."""
    problemas: list[str] = []
    advertencias: list[str] = []

    docs = leccion / "docs" / "es.md"
    code = leccion / "code"
    tests = code / "tests"
    outputs = leccion / "outputs"

    if not docs.exists():
        problemas.append("Falta docs/es.md")

    if not code.exists():
        problemas.append("Falta carpeta code/")
    elif not archivos_codigo(code):
        problemas.append("Falta archivo main.<lang> en code/")

    if not tests.exists():
        problemas.append("Falta carpeta code/tests/")
    else:
        n_pruebas = contar_pruebas(tests)
        if n_pruebas < MIN_PRUEBAS:
            problemas.append(
                f"Solo {n_pruebas} pruebas; mínimo {MIN_PRUEBAS}"
            )

    if not outputs.exists():
        advertencias.append("Sin outputs/ (opcional)")

    return {
        "leccion": leccion.name,
        "ruta": str(leccion.relative_to(RAIZ)),
        "ok": not problemas,
        "problemas": problemas,
        "advertencias": advertencias,
    }


def auditar_fase(fase_dir: Path) -> dict:
    lecciones = lecciones_de_fase(fase_dir)
    resultados = [auditar_leccion(p) for p in lecciones]
    return {
        "fase": fase_dir.name,
        "ruta": str(fase_dir.relative_to(RAIZ)),
        "n_lecciones": len(lecciones),
        "n_ok": sum(1 for r in resultados if r["ok"]),
        "lecciones": resultados,
    }


def auditar_repositorio(solo_fase: str | None = None) -> dict:
    if not FASES.exists():
        return {"error": f"No existe la carpeta {FASES}"}

    fases = sorted(p for p in FASES.iterdir() if p.is_dir() and FASE_ID.match(p.name))
    if solo_fase:
        fases = [p for p in fases if p.name.startswith(f"{solo_fase.zfill(2)}-")]

    resultados = [auditar_fase(p) for p in fases]
    return {
        "n_fases": len(fases),
        "n_lecciones_total": sum(r["n_lecciones"] for r in resultados),
        "n_lecciones_ok": sum(r["n_ok"] for r in resultados),
        "fases": resultados,
    }


def imprimir_texto(reporte: dict) -> None:
    print(f"Fases auditadas:           {reporte['n_fases']}")
    print(f"Lecciones totales:         {reporte['n_lecciones_total']}")
    print(f"Lecciones con estructura OK: {reporte['n_lecciones_ok']}")
    print()
    for fase in reporte["fases"]:
        estado = "✅" if fase["n_ok"] == fase["n_lecciones"] else "🚧"
        print(f"{estado} {fase['fase']}  ({fase['n_ok']}/{fase['n_lecciones']})")
        for lec in fase["lecciones"]:
            marca = "  ✅" if lec["ok"] else "  ❌"
            print(f"{marca} {lec['ruta']}")
            for p in lec["problemas"]:
                print(f"      - {p}")
            for a in lec["advertencias"]:
                print(f"      ~ {a}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fase", help="Limitar la auditoría a una fase (ej. 02).")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    reporte = auditar_repositorio(solo_fase=args.fase)

    if args.json:
        print(json.dumps(reporte, indent=2, ensure_ascii=False))
    else:
        imprimir_texto(reporte)

    total = reporte.get("n_lecciones_total", 0)
    ok = reporte.get("n_lecciones_ok", 0)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
