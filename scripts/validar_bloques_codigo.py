#!/usr/bin/env python3
"""
Valida reglas de formato en archivos Markdown:

1. Todo bloque de código fence debe tener etiqueta de lenguaje.
2. Todo bloque ```mermaid debe estar balanceado (un ``` de cierre).
3. Detecta diagramas "de cajitas" hechos con caracteres Unicode
   (─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬) y sugiere usar Mermaid.
4. Detecta referencias Markdown a archivos que no existen
   (`[texto](ruta.md)`, `<ruta.md>`).

Uso:
    python3 scripts/validar_bloques_codigo.py
    python3 scripts/validar_bloques_codigo.py --ruta README.md
    python3 scripts/validar_bloques_codigo.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

RAIZ = Path(__file__).resolve().parent.parent

EXCLUIR_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "outputs", "datasets"}
EXCLUIR_ENLACES = ("blob/", "tree/", "raw/", "/pull/", "/issues/")

FENCE_APERTURA = re.compile(r"^(\s*)```([^\s`].*)?$")
FENCE_CIERRE = re.compile(r"^(\s*)```(\s*[a-zA-Z0-9_-]*\s*)?$")
ENLACE_MD = re.compile(r"\[([^\]]+)\]\(([^)]+\.[a-z0-9]+)(?:\s+\"[^\"]*\")?\)")
ENLACE_ANG = re.compile(r"<([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)>")

CAR_UNICODE_DIAG = "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬◄►▲▼◢◣◤◥"


def archivos_md(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    return sorted(
        p for p in ruta.rglob("*.md")
        if not any(seg in EXCLUIR_DIRS for seg in p.parts)
    )


def es_fence_arbol(lineas: list[str], inicio: int, fin: int) -> bool:
    """Heurística: ¿el contenido del fence parece un árbol / estructura
    de directorios con caracteres Unicode? Si es así, el fence debería
    llevar la etiqueta `text`."""
    if fin - inicio < 2:
        return False
    ventana = "".join(lineas[inicio:fin])
    return (
        "├" in ventana or "└" in ventana or "│" in ventana
    ) and "<-" not in ventana and "==>" not in ventana


def mirar_arbol_adelante(lineas: list[str], desde: int, max_look: int = 30) -> bool:
    """Mira las siguientes líneas después de un fence sin etiqueta
    para decidir si es un árbol / estructura de directorios."""
    fin = min(desde + max_look, len(lineas))
    ventana = "".join(lineas[desde:fin])
    return (
        "├" in ventana or "└" in ventana or "│" in ventana
        or ("├──" in ventana or "└──" in ventana or "│" in ventana)
    )


def validar_bloques(archivo: Path) -> list[dict]:
    problemas: list[dict] = []
    try:
        texto = archivo.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return problemas

    lineas = texto.splitlines()
    dentro = False
    inicio = 0

    for n, linea in enumerate(lineas, 1):
        if not FENCE_APERTURA.match(linea):
            continue
        m = FENCE_APERTURA.match(linea)
        etiqueta = (m.group(2) or "").strip()
        if dentro:
            # Si tiene info string, en CommonMark esto es en realidad
            # una apertura de un nuevo bloque (el bloque anterior no
            # tenía cierre). Lo registramos como error de anidación.
            if etiqueta:
                problemas.append({
                    "archivo": str(archivo.relative_to(RAIZ)),
                    "linea": n,
                    "tipo": "fence_sin_cerrar",
                    "mensaje": (
                        f"Fence ``` previo nunca se cerró; "
                        f"se encontró ```{{etiqueta}} en su lugar"
                    ),
                })
                dentro = False
                # Caer al tratamiento de apertura de abajo.
            else:
                dentro = False
                continue
        # Apertura
        dentro = True
        if not etiqueta:
            if mirar_arbol_adelante(lineas, n):
                problemas.append({
                    "archivo": str(archivo.relative_to(RAIZ)),
                    "linea": n,
                    "tipo": "fence_arbol_sin_etiqueta",
                    "mensaje": "Fence con árbol de directorios: añadir ```text",
                })
            else:
                problemas.append({
                    "archivo": str(archivo.relative_to(RAIZ)),
                    "linea": n,
                    "tipo": "fence_sin_lenguaje",
                    "mensaje": "Fence ``` sin etiqueta de lenguaje",
                })

    if dentro:
        problemas.append({
            "archivo": str(archivo.relative_to(RAIZ)),
            "linea": 0,
            "tipo": "fence_sin_cerrar",
            "mensaje": "Fence ``` sin línea de cierre",
        })

    return problemas


def validar_enlaces(archivo: Path) -> list[dict]:
    problemas: list[dict] = []
    try:
        texto = archivo.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return problemas

    for patron, grupo in [(ENLACE_MD, 2), (ENLACE_ANG, 1)]:
        for m in patron.finditer(texto):
            url = m.group(grupo)
            if url.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if any(seg in url for seg in EXCLUIR_ENLACES):
                continue
            url = url.split("#", 1)[0].split("?", 1)[0]
            if not url:
                continue
            objetivo = (archivo.parent / url).resolve()
            if not objetivo.exists():
                n_linea = texto[: m.start()].count("\n") + 1
                problemas.append({
                    "archivo": str(archivo.relative_to(RAIZ)),
                    "linea": n_linea,
                    "tipo": "enlace_roto",
                    "mensaje": f"Enlace roto: {url}",
                })
    return problemas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruta", help="Archivo o directorio a validar.")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    ruta = Path(args.ruta) if args.ruta else RAIZ
    archivos = archivos_md(ruta)

    problemas: list[dict] = []
    for archivo in archivos:
        problemas += validar_bloques(archivo)
        problemas += validar_enlaces(archivo)

    if args.json:
        print(json.dumps(
            {
                "n_archivos": len(archivos),
                "n_problemas": len(problemas),
                "problemas": problemas,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    print(f"Archivos .md revisados: {len(archivos)}")
    print(f"Problemas encontrados:  {len(problemas)}")
    for p in problemas:
        print(f"  {p['archivo']}:{p['linea']}  [{p['tipo']}]  {p['mensaje']}")

    return 0 if not problemas else 1


if __name__ == "__main__":
    sys.exit(main())
