"""
Lección: 03-gpu-nube
Fase: 00
Prerrequisitos: 01-entorno-desarrollo
Fuentes:
- PyTorch CUDA: https://pytorch.org/docs/stable/notes/cuda.html
- MPS backend: https://pytorch.org/docs/stable/notes/mps.html
- platform: https://docs.python.org/3/library/platform.html
"""
from __future__ import annotations

import json
import platform
import sys
from typing import Optional


def detectar_cuda() -> Optional[dict]:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "backend": "cuda",
        "dispositivo": torch.cuda.get_device_name(0),
        "cantidad": torch.cuda.device_count(),
        "version_cuda": torch.version.cuda,
    }


def detectar_mps() -> Optional[dict]:
    try:
        import torch
    except Exception:
        return None
    if not torch.backends.mps.is_available():
        return None
    if not torch.backends.mps.is_built():
        return None
    return {
        "backend": "mps",
        "dispositivo": "Apple Silicon GPU",
        "cantidad": 1,
    }


def detectar_cpu() -> dict:
    return {
        "backend": "cpu",
        "dispositivo": platform.processor() or "CPU",
        "cantidad": 1,
    }


def seleccionar_backend() -> dict:
    info = detectar_cuda() or detectar_mps() or detectar_cpu()
    if info["backend"] == "cuda":
        info["recomendacion"] = "entrenar local en GPU NVIDIA"
    elif info["backend"] == "mps":
        info["recomendacion"] = "entrenar local con Apple Silicon"
    else:
        info["recomendacion"] = "usar Colab o Kaggle para GPU gratuita"
    return info


def main() -> int:
    info = seleccionar_backend()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
