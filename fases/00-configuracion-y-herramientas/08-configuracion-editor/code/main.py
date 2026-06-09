"""
Lección: 08-configuracion-editor
Fase: 00
Prerrequisitos: 01-entorno-desarrollo
Fuentes:
- VS Code settings: https://code.visualstudio.com/docs/getstarted/settings
- VS Code extensions: https://code.visualstudio.com/docs/editor/extension-marketplace
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXTENSIONES_RECOMENDADAS = {
    "ms-python.python": "Python",
    "ms-python.vscode-pylance": "Pylance",
    "ms-toolsai.jupyter": "Jupyter",
    "ms-toolsai.vscode-jupyter": "Jupyter Notebooks",
    "ms-azuretools.vscode-docker": "Docker",
    "eamodio.gitlens": "GitLens",
}


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def validar_settings(datos: dict) -> list[str]:
    problemas: list[str] = []
    if "python.defaultInterpreterPath" not in datos:
        problemas.append("Falta 'python.defaultInterpreterPath'")
    if not datos.get("editor.formatOnSave", False):
        problemas.append("editor.formatOnSave no esta habilitado")
    if "[python]" in datos and "editor.defaultFormatter" in datos["[python]"]:
        return problemas
    problemas.append("Falta [python].editor.defaultFormatter")
    return problemas


def validar_extensions(datos: dict) -> list[str]:
    problemas: list[str] = []
    recomendaciones = datos.get("recommendations", [])
    for ext_id in ("ms-python.python", "ms-python.vscode-pylance"):
        if ext_id not in recomendaciones:
            problemas.append(f"Falta extension recomendada: {ext_id}")
    return problemas


def verificar_directorio(vscode: Path) -> dict:
    reporte: dict = {
        "settings_ok": True,
        "extensions_ok": True,
        "problemas": [],
    }
    settings = vscode / "settings.json"
    extensions = vscode / "extensions.json"
    if settings.exists():
        try:
            datos = cargar_json(settings)
            for p in validar_settings(datos):
                reporte["problemas"].append(f"settings: {p}")
                reporte["settings_ok"] = False
        except json.JSONDecodeError as exc:
            reporte["problemas"].append(f"settings: JSON invalido: {exc}")
            reporte["settings_ok"] = False
    else:
        reporte["problemas"].append("no existe settings.json")
        reporte["settings_ok"] = False
    if extensions.exists():
        try:
            datos = cargar_json(extensions)
            for p in validar_extensions(datos):
                reporte["problemas"].append(f"extensions: {p}")
                reporte["extensions_ok"] = False
        except json.JSONDecodeError as exc:
            reporte["problemas"].append(f"extensions: JSON invalido: {exc}")
            reporte["extensions_ok"] = False
    else:
        reporte["problemas"].append("no existe extensions.json")
        reporte["extensions_ok"] = False
    return reporte


def main() -> int:
    vscode = Path(".vscode")
    if not vscode.exists():
        print(f"AVISO: {vscode} no existe; crea los archivos ahi")
        return 1
    reporte = verificar_directorio(vscode)
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    return 0 if not reporte["problemas"] else 1


if __name__ == "__main__":
    sys.exit(main())
