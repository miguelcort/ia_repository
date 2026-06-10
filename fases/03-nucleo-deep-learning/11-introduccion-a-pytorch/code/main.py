"""
Lección: 11-introduccion-a-pytorch
Fase: 03
Prerrequisitos: 10-mini-framework
"""
from __future__ import annotations
import sys
import numpy as np


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        disponible = True
    except ImportError:
        disponible = False
    if not disponible:
        print("PyTorch no esta instalado. Esta leccion requiere:")
        print("  pip install torch numpy")
        print("Conceptos cubiertos (sin codigo ejecutable):")
        print("  - Tensor: ndarray con gradiente automatico")
        print("  - nn.Module: clase base para modelos")
        print("  - nn.Linear, nn.ReLU, nn.Conv2d, etc.")
        print("  - loss: nn.MSELoss, nn.CrossEntropyLoss")
        print("  - optim: torch.optim.SGD, Adam, AdamW")
        print("  - DataLoader para batches")
        return 0

    # Ejemplo ejecutable: red pequena con autograd
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    modelo = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(modelo.parameters(), lr=0.05)

    for epoca in range(500):
        y_pred = modelo(X)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        pred = modelo(X)
    print(f"XOR PyTorch: {pred.flatten().round(3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())