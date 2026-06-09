"""
Lección: 01-entorno-desarrollo
Fase: 00
Prerrequisitos: Ninguno
Fuentes:
- Python: https://docs.python.org/3/library/sys.html
- PyTorch: https://pytorch.org/docs/stable/notes/cuda.html
- shutil: https://docs.python.org/3/library/shutil.html
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from typing import Callable, Optional

CheckResult = tuple[str, str, Optional[str]]


def check(name: str, predicate: Callable[[], bool], detail: Callable[[], str] | None = None) -> CheckResult:
    try:
        ok = bool(predicate())
    except Exception as exc:
        return (name, "fail", f"{type(exc).__name__}: {exc}")
    if not ok:
        return (name, "fail", None)
    info = detail() if detail else None
    return (name, "pass", info)


def check_python() -> CheckResult:
    major, minor = sys.version_info[:2]
    passed = (major, minor) >= (3, 10)
    if not passed:
        return ("python", "fail", f"{major}.{minor}")
    return ("python", "pass", f"{major}.{minor}.{sys.version_info.micro}")


def check_import(module: str) -> CheckResult:
    try:
        mod = __import__(module)
    except Exception as exc:
        return (module, "fail", str(exc))
    ver = getattr(mod, "__version__", "desconocida")
    return (module, "pass", f"v{ver}")


def check_command(binary: str) -> CheckResult:
    path = shutil.which(binary)
    if path is None:
        return (binary, "fail", None)
    return (binary, "pass", path)


def check_pytorch_cuda() -> CheckResult:
    try:
        import torch
    except Exception:
        return ("torch-cuda", "warn", "PyTorch no instalado")
    try:
        disponible = bool(torch.cuda.is_available())
    except Exception as exc:
        return ("torch-cuda", "fail", str(exc))
    if not disponible:
        return ("torch-cuda", "warn", "CPU only (CUDA no disponible)")
    nombre = torch.cuda.get_device_name(0)
    return ("torch-cuda", "pass", nombre)


def verificar_entorno() -> dict:
    checks: list[CheckResult] = [
        check_python(),
        check_import("numpy"),
        check_command("git"),
    ]
    try:
        checks.append(check_import("torch"))
        checks.append(check_pytorch_cuda())
    except Exception:
        pass
    return {
        "sistema": platform.platform(),
        "python": sys.version.split()[0],
        "checks": [
            {"nombre": n, "estado": e, "detalle": d}
            for n, e, d in checks
        ],
    }


def main() -> int:
    reporte = verificar_entorno()
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    fallaron = [c for c in reporte["checks"] if c["estado"] == "fail"]
    if fallaron:
        print(f"\nATENCIÓN: {len(fallaron)} checks fallaron.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
