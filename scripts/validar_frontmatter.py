#!/usr/bin/env python3
"""
Valida el frontmatter de `docs/es.md` en cada lección y la
nomenclatura de las carpetas y archivos.

Reglas:

1. Toda carpeta de lección debe llamarse `NN-slug-kebab-case` con `NN`
   de dos dígitos (00-99).
2. Toda carpeta de fase debe llamarse `NN-nombre-fase`.
3. El archivo `docs/es.md` debe existir y tener frontmatter con:
     - `**Tipo:**` Learn | Construir | Referencia
     - `**Lenguajes:**` lista no vacía
     - `**Prerrequisitos:**` (cadena no vacía o "Ninguno")
     - `**Tiempo estimado:**` ~N minutos
4. El código debe estar en `code/main.<lang>` con un
   `code/tests/test_main.<lang>` que tenga al menos 5 pruebas.
5. Los nombres de archivos y carpetas sólo pueden contener
   `[a-z0-9-]`, sin acentos, espacios ni mayúsculas.

Uso:
    python3 scripts/validar_frontmatter.py
    python3 scripts/validar_frontmatter.py --fase 02
    python3 scripts/validar_frontmatter.py --json
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

LECCION_ID = re.compile(r"^(\d{2})-([a-z0-9-]+)$")
FASE_ID = re.compile(r"^(\d{2})-([a-z0-9-]+)$")
ARCHIVO_ID = re.compile(r"^[a-z0-9_-]+\.[a-z0-9]+$")

TIPOS_VALIDOS = {"Learn", "Aprender", "Construir", "Build", "Referencia", "Reference"}
LENG_VALIDOS = {
    "python", "py", "typescript", "ts", "rust", "rs",
    "julia", "jl", "bash", "sh", "shell", "docker", "yaml", "yml", "json",
    "markdown", "md", "html", "css", "javascript", "js", "tsx", "jsx",
    "r", "sql", "text",
}

PRUEBAS = {
    ".py": re.compile(r"^\s*def\s+test_\w+", re.M),
    ".rs": re.compile(r"#\[test\]"),
    ".jl": re.compile(r"@test\s+\w+\s*="),
    ".ts": re.compile(r"(it|test)\s*\(", re.M),
}


def es_nombre_valido(nombre: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9]", nombre))


def lecciones_de_fase(fase_dir: Path) -> list[Path]:
    return sorted(
        p for p in fase_dir.iterdir()
        if p.is_dir() and p.name[:2].isdigit()
    )


def archivos_codigo(code_dir: Path) -> list[Path]:
    if not code_dir.exists():
        return []
    return [
        p for p in code_dir.iterdir()
        if p.is_file() and p.suffix in {".py", ".ts", ".rs", ".jl"}
    ]


def contar_pruebas(test_dir: Path) -> int:
    if not test_dir.exists():
        return 0
    total = 0
    for p in test_dir.rglob("*"):
        if not p.is_file():
            continue
        patron = PRUEBAS.get(p.suffix)
        if not patron:
            continue
        try:
            texto = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total += len(patron.findall(texto))
    return total


def validar_leccion(leccion: Path) -> dict:
    problemas: list[str] = []
    advertencias: list[str] = []

    if not es_nombre_valido(leccion.name):
        problemas.append(f"Nombre de lección inválido: {leccion.name!r}")

    m = LECCION_ID.match(leccion.name)
    if m:
        lec_num = int(m.group(1))
        fase_num = int(leccion.parent.name[:2])
        if lec_num > 99:
            problemas.append(f"Número de lección {lec_num} fuera de rango (00-99)")

    docs = leccion / "docs" / "es.md"
    if not docs.exists():
        problemas.append("Falta docs/es.md")
    else:
        texto = docs.read_text(encoding="utf-8", errors="ignore")
        problemas += validar_frontmatter(texto)

    code = leccion / "code"
    mains = archivos_codigo(code)
    if not mains:
        problemas.append("Falta code/main.<lang>")
    else:
        for p in code.rglob("*"):
            if p.is_file() and not es_nombre_valido(p.name):
                advertencias.append(f"Nombre de archivo no kebab-case: {p.name}")

    tests = code / "tests"
    if not tests.exists():
        problemas.append("Falta code/tests/")
    else:
        n = contar_pruebas(tests)
        if n < 5:
            problemas.append(f"Solo {n} pruebas (mínimo 5)")

    return {
        "leccion": leccion.name,
        "ruta": str(leccion.relative_to(RAIZ)),
        "ok": not problemas,
        "problemas": problemas,
        "advertencias": advertencias,
    }


def validar_frontmatter(texto: str) -> list[str]:
    problemas: list[str] = []
    campos_obligatorios = [
        r"\*\*Tipo:\*\*\s*([^\n]+)",
        r"\*\*Lenguajes:\*\*\s*([^\n]+)",
        r"\*\*Prerrequisitos:\*\*\s*([^\n]+)",
        r"\*\*Tiempo estimado:\*\*\s*([^\n]+)",
    ]
    for patron in campos_obligatorios:
        m = re.search(patron, texto)
        if not m:
            problemas.append(f"Falta frontmatter: {patron.split('*')[2].strip(':')}")
            continue
        valor = m.group(1).strip()
        if patron.startswith(r"\*\*Tipo"):
            tipo = valor.split()[0] if valor else ""
            if tipo not in {t for t in TIPOS_VALIDOS}:
                problemas.append(
                    f"Tipo desconocido: {tipo!r} (esperado Learn, Construir o Referencia)"
                )
        elif patron.startswith(r"\*\*Lenguajes"):
            lenguajes = {l.strip().lower() for l in re.split(r"[,/]", valor) if l.strip()}
            invalidos = lenguajes - LENG_VALIDOS
            if invalidos:
                problemas.append(f"Lenguajes no reconocidos: {sorted(invalidos)}")
    return problemas


def validar_fase(fase: Path) -> dict:
    problemas: list[str] = []
    if not es_nombre_valido(fase.name):
        problemas.append(f"Nombre de fase inválido: {fase.name!r}")
    if not (fase / "README.md").exists():
        problemas.append("Falta README.md de la fase")

    lecciones = lecciones_de_fase(fase)
    resultados = [validar_leccion(p) for p in lecciones]
    return {
        "fase": fase.name,
        "n_lecciones": len(lecciones),
        "n_ok": sum(1 for r in resultados if r["ok"]),
        "problemas_fase": problemas,
        "lecciones": resultados,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fase", help="Limitar a una fase (ej. 02).")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    fases = sorted(
        p for p in FASES.iterdir()
        if p.is_dir() and FASE_ID.match(p.name)
    )
    if args.fase:
        fases = [p for p in fases if p.name.startswith(f"{int(args.fase):02d}-")]

    resultados = [validar_fase(p) for p in fases]

    if args.json:
        print(json.dumps(
            {
                "n_fases": len(fases),
                "fases": resultados,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    total_ok = sum(f["n_ok"] for f in resultados)
    total = sum(f["n_lecciones"] for f in resultados)
    print(f"Fases: {len(fases)} · Lecciones: {total} · OK: {total_ok}")
    for f in resultados:
        marca = "✅" if f["n_ok"] == f["n_lecciones"] and not f["problemas_fase"] else "🚧"
        print(f"{marca} {f['fase']}  ({f['n_ok']}/{f['n_lecciones']})")
        for p in f["problemas_fase"]:
            print(f"    - {p}")
        for l in f["lecciones"]:
            for p in l["problemas"]:
                print(f"      ❌ {l['leccion']}: {p}")

    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
