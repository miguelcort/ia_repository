#!/usr/bin/env python3
"""
Ejecuta todas las validaciones y muestra un resumen consolidado.

Uso:
    python3 scripts/validar_todo.py
    python3 scripts/validar_todo.py --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SCRIPTS = RAIZ / "scripts"

VALIDADORES = [
    ("auditar_lecciones.py", "Auditoría estructural de fases y lecciones"),
    ("validar_frontmatter.py", "Validar frontmatter y nombres de archivos"),
    ("validar_bloques_codigo.py", "Validar bloques de código y enlaces"),
    ("actualizar_conteo.py", "Sincronizar conteo de lecciones"),
]


def correr(nombre: str, descripcion: str, args: list[str]) -> dict:
    script = SCRIPTS / nombre
    print(f"\n=== {descripcion} ===")
    print(f"    $ python3 {nombre} {' '.join(args)}")
    resultado = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    print(resultado.stdout)
    if resultado.stderr:
        print("STDERR:", resultado.stderr)
    return {
        "script": nombre,
        "descripcion": descripcion,
        "returncode": resultado.returncode,
        "ok": resultado.returncode == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    resultados = [
        correr(nombre, desc, [])
        for nombre, desc in VALIDADORES
    ]

    if args.json:
        print(json.dumps(resultados, indent=2, ensure_ascii=False))
    else:
        ok = sum(1 for r in resultados if r["ok"])
        print(f"\n=== Resumen: {ok}/{len(resultados)} validadores OK ===")
        for r in resultados:
            marca = "✅" if r["ok"] else "❌"
            print(f"  {marca} {r['script']}  (exit {r['returncode']})")

    return 0 if all(r["ok"] for r in resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
