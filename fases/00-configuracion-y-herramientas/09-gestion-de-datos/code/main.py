"""
Lección: 09-gestion-de-datos
Fase: 00
Prerrequisitos: 01-entorno-desarrollo, 02-git-colaboracion
Fuentes:
- DVC: https://dvc.org/doc
- hashlib: https://docs.python.org/3/library/hashlib.html
- csv: https://docs.python.org/3/library/csv.html
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cargar_csv(ruta: Path) -> tuple[list[str], list[list[str]]]:
    with ruta.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        cabecera = next(reader, [])
        filas = list(reader)
    return cabecera, filas


def validar_no_vacio(cabecera: list[str], filas: list[list[str]]) -> list[str]:
    problemas: list[str] = []
    if not cabecera:
        problemas.append("CSV sin cabecera")
    if not filas:
        problemas.append("CSV sin filas")
    for i, fila in enumerate(filas):
        if len(fila) != len(cabecera):
            problemas.append(f"fila {i}: {len(fila)} columnas != {len(cabecera)}")
            if len(problemas) > 10:
                problemas.append("demasiados errores, abortando")
                break
    return problemas


def generar_manifiesto(ruta: Path) -> dict:
    cabecera, filas = cargar_csv(ruta)
    return {
        "archivo": str(ruta),
        "hash_sha256": sha256_archivo(ruta),
        "tamano_bytes": ruta.stat().st_size,
        "filas": len(filas),
        "columnas": cabecera,
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Uso: python3 main.py <dataset.csv>")
        return 1
    ruta = Path(argv[0])
    if not ruta.exists():
        print(f"ERROR: {ruta} no existe")
        return 1
    cabecera, filas = cargar_csv(ruta)
    problemas = validar_no_vacio(cabecera, filas)
    if problemas:
        print("Problemas:", *problemas, sep="\n  - ")
        return 1
    manifiesto = generar_manifiesto(ruta)
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
