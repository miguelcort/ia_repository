"""
Lección: 11-descomposicion-svd
Fase: 01
Prerrequisitos: 03-transformaciones-valores-propios
"""
from __future__ import annotations
import sys
import numpy as np


def svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(A, full_matrices=False)


def reconstruccion(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def main() -> int:
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    U, S, Vt = svd(A)
    print(f"U shape={U.shape}, S={S}, Vt shape={Vt.shape}")
    for k in [1, 2, 3]:
        A_k = reconstruccion(U, S, Vt, k)
        err = np.linalg.norm(A - A_k)
        print(f"  k={k}: error={err:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())