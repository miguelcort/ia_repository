#!/usr/bin/env python3
"""
Sincroniza el conteo de lecciones en README.md y ROADMAP.md con la
realidad del repositorio (carpetas `fases/NN-.../MM-...`).

Para cada fase cuenta las carpetas de lección que existan, luego actualiza:

- README.md — la tabla de 20 fases (columna *Estado*).
- ROADMAP.md — la tabla de 20 fases (columna *Estado* y conteos
  aproximados).

Estados:

- `⬚` (pendiente) si la fase no tiene carpeta de lecciones.
- `🚧` (en construcción) si la fase tiene al menos una lección.
- `✅` (completa) si todas las lecciones planeadas ya existen.

El conteo *planeado* se obtiene del README.md actual; si una fase no
tiene lecciones planeadas explícitas, se asume 0.

Uso:

    # Mostrar el conteo sin tocar archivos
    python3 scripts/actualizar_conteo.py

    # Sobrescribir README.md y ROADMAP.md
    python3 scripts/actualizar_conteo.py --write

    # Sólo una fase
    python3 scripts/actualizar_conteo.py --fase 02
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FASES = RAIZ / "fases"
README = RAIZ / "README.md"
ROADMAP = RAIZ / "ROADMAP.md"

LECCION_ID = re.compile(r"^(\d{2})-")
FASE_ID = re.compile(r"^(\d{2})-(.+)$")
NUMERO_TABLA = re.compile(r"^\|\s*(\d{1,2})\s*\|")

ESTADO_GLYPH = {"✅": "✅", "🚧": "🚧", "⬚": "⬚"}


def lecciones_de_fase(fase_dir: Path) -> list[Path]:
    if not fase_dir.exists():
        return []
    return sorted(
        p for p in fase_dir.iterdir()
        if p.is_dir() and LECCION_ID.match(p.name)
    )


def lecciones_planeadas(texto: str) -> dict[int, int]:
    """Lee la tabla del ROADMAP para extraer el número de lecciones
    planeadas por fase. Devuelve {fase_int: n_lecciones_planeadas}."""
    planeadas: dict[int, int] = {}
    en_tabla = False
    for linea in texto.splitlines():
        if linea.startswith("| # | Fase | Slug |"):
            en_tabla = True
            continue
        if en_tabla and linea.startswith("|---"):
            continue
        if en_tabla and not linea.startswith("|"):
            en_tabla = False
        if not en_tabla:
            continue
        celdas = [c.strip() for c in linea.strip("|").split("|")]
        if len(celdas) < 5:
            continue
        try:
            fase_num = int(celdas[0])
        except ValueError:
            continue
        planeadas[fase_num] = celdas[3]
    return planeadas


def nueva_linea_tabla(celdas: list[str], glyph: str) -> str:
    """Sustituye la última celda de la fila (estado) por el glifo."""
    celdas = list(celdas)
    celdas[-1] = glyph
    return "| " + " | ".join(celdas) + " |"


def actualizar_tabla(texto: str, conteo: dict[int, int], planeadas: dict[int, int], *, write: bool) -> tuple[str, int]:
    """Recorre las líneas; en cada fila de la tabla de fases sustituye la
    columna de estado. Devuelve (texto_nuevo, n_cambios)."""
    lineas = texto.splitlines()
    nuevas: list[str] = []
    en_tabla = False
    cambios = 0

    for i, linea in enumerate(lineas):
        if "| # | Fase | Slug |" in linea or "| # | Fase | Slug |" in linea:
            en_tabla = True
            nuevas.append(linea)
            continue
        if en_tabla and linea.startswith("|---"):
            nuevas.append(linea)
            continue
        if en_tabla and not linea.startswith("|"):
            en_tabla = False
            nuevas.append(linea)
            continue

        if en_tabla and linea.startswith("|"):
            celdas = [c.strip() for c in linea.strip("|").split("|")]
            if len(celdas) < 5:
                nuevas.append(linea)
                continue
            try:
                fase_num = int(celdas[0])
            except ValueError:
                nuevas.append(linea)
                continue

            n_existentes = conteo.get(fase_num, 0)
            plan_str = planeadas.get(fase_num, "—")
            try:
                n_plan = int(plan_str)
            except (TypeError, ValueError):
                n_plan = 0

            if n_existentes == 0:
                glyph = "⬚"
            elif n_plan and n_existentes >= n_plan:
                glyph = "✅"
            else:
                glyph = "🚧"

            nueva = nueva_linea_tabla(celdas, glyph)
            if nueva != linea:
                cambios += 1
            nuevas.append(nueva)
        else:
            nuevas.append(linea)

    return ("\n".join(nuevas) + "\n", cambios)


def contar_fases() -> dict[int, int]:
    conteo: dict[int, int] = {}
    for fase in FASES.iterdir():
        m = FASE_ID.match(fase.name)
        if not m:
            continue
        fase_num = int(m.group(1))
        conteo[fase_num] = len(lecciones_de_fase(fase))
    return conteo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Escribir cambios en los archivos.")
    parser.add_argument("--fase", help="Limitar a una fase (ej. 02).")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    conteo = contar_fases()
    planeadas_roadmap = lecciones_planeadas(ROADMAP.read_text(encoding="utf-8"))
    planeadas_readme = lecciones_planeadas(README.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(
            {
                "conteo": conteo,
                "planeadas_roadmap": planeadas_roadmap,
                "planeadas_readme": planeadas_readme,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    print("Conteo real de lecciones por fase:")
    for fase in sorted(conteo):
        print(f"  Fase {fase:>2}: {conteo[fase]} lecciones")

    cambios_total = 0
    for archivo, planeadas in [(README, planeadas_readme), (ROADMAP, planeadas_roadmap)]:
        texto_original = archivo.read_text(encoding="utf-8")
        texto_nuevo, cambios = actualizar_tabla(texto_original, conteo, planeadas, write=args.write)
        cambios_total += cambios
        if args.write and cambios:
            archivo.write_text(texto_nuevo, encoding="utf-8")
            print(f"  → {archivo.name}: {cambios} filas actualizadas")
        else:
            print(f"  → {archivo.name}: {cambios} filas cambiadas (no se escribió, use --write)")

    cambios_fases = actualizar_leyendas_de_fase(conteo, write=args.write)
    if args.write:
        print(f"  → leyendas de fase: {cambios_fases} actualizadas")
    else:
        print(f"  → leyendas de fase: {cambios_fases} cambiadas (use --write)")

    return 0 if (cambios_total == 0 and cambios_fases == 0) or args.write else 1


def actualizar_leyendas_de_fase(conteo: dict[int, int], *, write: bool) -> int:
    """Actualiza la columna `Estado` de las tablas de lecciones dentro de
    cada README de fase: ✅ si la lección tiene carpeta en disco,
    🚧 en caso contrario. La columna objetivo es la segunda (índice 1)."""
    cambios = 0
    for fase_dir in sorted(FASES.iterdir()):
        m = FASE_ID.match(fase_dir.name)
        if not m:
            continue
        fase_num = int(m.group(1))
        readme = fase_dir / "README.md"
        if not readme.exists():
            continue
        existentes = {p.name for p in lecciones_de_fase(fase_dir)}
        texto = readme.read_text(encoding="utf-8")
        lineas = texto.splitlines()
        nuevas: list[str] = []
        en_tabla = False
        for linea in lineas:
            if "| # |" in linea and "Lección" in linea:
                en_tabla = True
                nuevas.append(linea)
                continue
            if en_tabla and linea.startswith("|---"):
                nuevas.append(linea)
                continue
            if en_tabla and not linea.startswith("|"):
                en_tabla = False
                nuevas.append(linea)
                continue
            if en_tabla and linea.startswith("|"):
                celdas = [c.strip() for c in linea.strip("|").split("|")]
                if len(celdas) < 2:
                    nuevas.append(linea)
                    continue
                m2 = LECCION_ID.match(celdas[0])
                if not m2:
                    nuevas.append(linea)
                    continue
                lec_slug = m2.group(0) + "-"
                glyph = "✅" if any(name.startswith(lec_slug) for name in existentes) else "🚧"
                celdas[1] = glyph
                nueva = "| " + " | ".join(celdas) + " |"
                if nueva != linea:
                    cambios += 1
                nuevas.append(nueva)
            else:
                nuevas.append(linea)
        if write and nuevas != lineas:
            readme.write_text("\n".join(nuevas) + "\n", encoding="utf-8")
    return cambios


if __name__ == "__main__":
    sys.exit(main())
