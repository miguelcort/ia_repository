"""
Lección: 04-apis-y-claves
Fase: 00
Prerrequisitos: 01-entorno-desarrollo, 02-git-colaboracion
Fuentes:
- os.environ: https://docs.python.org/3/library/os.html
- urllib: https://docs.python.org/3/library/urllib.request.html
- OpenAI auth: https://platform.openai.com/docs/api-reference/authentication
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLACEHOLDERS = {"", "your-key-here", "xxx", "changeme", "sk-xxx", "<your-key>"}


def cargar_env(archivo: Path = Path(".env")) -> int:
    if not archivo.exists():
        return 0
    cargados = 0
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        if clave and clave not in os.environ:
            os.environ[clave] = valor
            cargados += 1
    return cargados


def validar_clave(nombre: str, valor: str, prefijo: str = "") -> tuple[bool, str]:
    if not valor:
        return (False, f"{nombre}: vacia")
    if valor.lower() in PLACEHOLDERS:
        return (False, f"{nombre}: parece un placeholder ({valor!r})")
    if prefijo and not valor.startswith(prefijo):
        return (False, f"{nombre}: no empieza con el prefijo esperado {prefijo!r}")
    if len(valor) < 16:
        return (False, f"{nombre}: demasiado corta ({len(valor)} chars)")
    return (True, f"{nombre}: ok ({len(valor)} chars)")


def main() -> int:
    cargados = cargar_env()
    if cargados:
        print(f"[info] {cargados} variables cargadas desde .env")
    proveedores = {
        "OPENAI_API_KEY": "sk-",
        "ANTHROPIC_API_KEY": "sk-ant-",
        "HUGGINGFACE_TOKEN": "hf_",
    }
    resultados = {}
    for nombre, prefijo in proveedores.items():
        resultados[nombre] = validar_clave(nombre, os.environ.get(nombre, ""), prefijo)
    print(json.dumps(
        {n: {"ok": ok, "msg": msg} for n, (ok, msg) in resultados.items()},
        indent=2, ensure_ascii=False,
    ))
    fallaron = [n for n, (ok, _) in resultados.items() if not ok]
    if fallaron:
        print(f"\n[warn] {len(fallaron)} claves con problemas.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
