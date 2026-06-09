"""
Lección: 05-jupyter-notebooks
Fase: 00
Prerrequisitos: 01-entorno-desarrollo
Fuentes:
- nbformat: https://nbformat.readthedocs.io/
- Jupyter: https://jupyter.org/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATRON_IMPORT = re.compile(r"^(?:from\s+(\S+)\s+import|import\s+(\S+))")


def cargar_notebook(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def obtener_codigo(celda: dict) -> str:
    return "".join(celda.get("source", []))


def validar_estructura(nb: dict) -> list[str]:
    problemas: list[str] = []
    celdas = nb.get("cells", [])
    if not celdas:
        return ["notebook vacio"]
    primera = obtener_codigo(celdas[0])
    imports = PATRON_IMPORT.findall(primera)
    if not imports:
        problemas.append("celda 1: no parece tener imports")
    codigo = [c for c in celdas if c.get("cell_type") == "code"]
    if not codigo:
        problemas.append("sin celdas de codigo")
    texto_total = "".join(obtener_codigo(c) for c in codigo)
    if "random" in texto_total or "numpy" in texto_total:
        if "seed" not in texto_total and "semilla" not in texto_total.lower():
            problemas.append("usa random/numpy pero no fija semilla")
    return problemas


def contar_palabras_markdown(nb: dict) -> int:
    total = 0
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            total += len(" ".join(c.get("source", [])).split())
    return total


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Uso: python3 main.py <notebook.ipynb>")
        return 1
    ruta = Path(argv[0])
    if not ruta.exists():
        print(f"ERROR: {ruta} no existe")
        return 1
    nb = cargar_notebook(ruta)
    problemas = validar_estructura(nb)
    palabras = contar_palabras_markdown(nb)
    reporte = {
        "archivo": str(ruta),
        "celdas_totales": len(nb.get("cells", [])),
        "palabras_markdown": palabras,
        "problemas": problemas,
    }
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    return 0 if not problemas else 1


if __name__ == "__main__":
    sys.exit(main())
