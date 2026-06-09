"""
Lección: 12-operaciones-con-tensores
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
"""
from __future__ import annotations
import sys
import numpy as np


def shape(t):
    return t.shape


def ndim(t):
    return t.ndim


def reshape(t, new_shape):
    return t.reshape(new_shape)


def transpose(t, axes=None):
    return np.transpose(t, axes=axes)


def broadcast(a, b):
    return a + b


def matmul(a, b):
    return a @ b


def einsum(espec, *args):
    return np.einsum(espec, *args)


def main() -> int:
    # Escalar
    s = np.array(3.14)
    # Vector
    v = np.array([1, 2, 3])
    # Matriz
    M = np.array([[1, 2], [3, 4]])
    # Tensor 3D
    T = np.arange(24).reshape(2, 3, 4)
    print(f"Escalar: shape={shape(s)}, ndim={ndim(s)}")
    print(f"Vector: shape={shape(v)}, ndim={ndim(v)}")
    print(f"Matriz: shape={shape(M)}, ndim={ndim(M)}")
    print(f"Tensor 3D: shape={shape(T)}, ndim={ndim(T)}")
    # Reshape
    print(f"Tensor 3D reshape (6, 4): {reshape(T, (6, 4)).shape}")
    # Transpose
    print(f"Transpose T (0,2,1): {transpose(T, (0, 2, 1)).shape}")
    # Broadcast
    a = np.array([[1], [2], [3]])
    b = np.array([10, 20, 30])
    print(f"Broadcast (3,1) + (3,) = {broadcast(a, b).shape}")
    # Matmul batch
    A = np.random.default_rng(0).random((5, 3, 4))
    B = np.random.default_rng(0).random((5, 4, 2))
    print(f"Batch matmul: {matmul(A, B).shape}")
    # Einsum
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print(f"einsum('i,i->', x, y) = {einsum('i,i->', x, y)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())