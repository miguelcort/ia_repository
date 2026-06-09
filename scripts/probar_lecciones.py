#!/usr/bin/env python3
"""
Ejecuta los tests de cada lección que tenga `code/tests/test_main.py`.

Por defecto, sólo corre las lecciones indicadas en la línea de comandos
(slug o ruta) o las que se hayan modificado. Si no se pasa nada, corre
todas.

Exit codes:
    0 — todas las pruebas pasaron
    1 — al menos una lección falló

Uso:
    python3 scripts/probar_lecciones.py                # todas
    python3 scripts/probar_lecciones.py --fase 00     # sólo fase 0
    python3 scripts/probar_lecciones.py \
        fases/00-configuracion-y-herramientas/01-entorno-desarrollo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FASES = RAIZ / "fases"
TIMEOUT = 30  # segundos por lección


def lecciones_objetivo(args: argparse.Namespace) -> list[Path]:
    """Devuelve la lista de carpetas de lección a probar."""
    if args.rutas:
        out: list[Path] = []
        for r in args.rutas:
            p = (RAIZ / r).resolve()
            if not p.exists():
                sys.exit(f"No existe: {p}")
            if (p / "code" / "tests" / "test_main.py").exists():
                out.append(p)
        return sorted(out)
    if args.fase:
        out = []
        for f in FASES.iterdir():
            if f.name.startswith(f"{int(args.fase):02d}-"):
                for l in f.iterdir():
                    if (l / "code" / "tests" / "test_main.py").exists():
                        out.append(l)
        return sorted(out)
    out = []
    for f in FASES.iterdir():
        for l in f.iterdir():
            if l.is_dir() and (l / "code" / "tests" / "test_main.py").exists():
                out.append(l)
    return sorted(out)


def probar_leccion(leccion: Path) -> tuple[str, float, str]:
    """Ejecuta los tests de una lección. Devuelve (estado, duracion, salida)."""
    inicio = time.time()
    try:
        resultado = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=leccion / "code",
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", time.time() - inicio, f"Timeout después de {TIMEOUT}s")
    salida = (resultado.stdout + resultado.stderr).strip()
    estado = "OK" if resultado.returncode == 0 else f"FAIL({resultado.returncode})"
    return (estado, time.time() - inicio, salida)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fase", help="Limitar a una fase (ej. 00).")
    parser.add_argument("rutas", nargs="*", help="Rutas específicas a probar.")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    parser.add_argument("--quiet", action="store_true", help="No imprimir la salida de cada test.")
    args = parser.parse_args(argv)

    lecciones = lecciones_objetivo(args)
    if not lecciones:
        print("No hay lecciones con tests para probar.")
        return 0

    if not args.json:
        print(f"Probando {len(lecciones)} lecciones...\n")

    resultados: list[dict] = []
    ok = 0
    for lec in lecciones:
        estado, duracion, salida = probar_leccion(lec)
        slug = lec.relative_to(RAIZ).as_posix()
        resultados.append({"leccion": slug, "estado": estado, "duracion_s": round(duracion, 2)})
        if estado == "OK":
            ok += 1
        if not args.json:
            marca = "✅" if estado == "OK" else "❌"
            print(f"  {marca} {slug}  ({duracion:.1f}s)  [{estado}]")
            if not args.quiet and estado != "OK":
                print(f"    {salida[-500:]}")

    if args.json:
        import json
        print(json.dumps({"n_ok": ok, "n_total": len(lecciones), "resultados": resultados}, indent=2, ensure_ascii=False))
    else:
        print(f"\nResumen: {ok}/{len(lecciones)} lecciones OK")

    return 0 if ok == len(lecciones) else 1


if __name__ == "__main__":
    sys.exit(main())
