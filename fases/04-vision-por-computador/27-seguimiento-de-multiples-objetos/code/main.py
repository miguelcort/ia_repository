"""
Lección: 27-seguimiento-de-multiples-objetos
Fase: 04
Prerrequisitos: 06-deteccion-de-objetos-yolo
"""
from __future__ import annotations
import sys
import numpy as np


def iou_box(a, b):
    """IoU entre dos bboxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def iou_matrix(boxes_a, boxes_b):
    """Matriz IoU entre dos listas de bboxes. shape (N, M)."""
    N = len(boxes_a)
    M = len(boxes_b)
    mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            mat[i, j] = iou_box(boxes_a[i], boxes_b[j])
    return mat


def hungarian_assignment(cost_matrix):
    """Asignacion hungara (mock: greedy por minimo).
    En produccion, usar scipy.optimize.linear_sum_assignment."""
    if cost_matrix.size == 0:
        return []
    N, M = cost_matrix.shape
    pares = []
    disponibles = list(range(M))
    for i in range(N):
        if not disponibles:
            break
        # Mejor match disponible
        costos = [(cost_matrix[i, j], j) for j in disponibles]
        costos.sort()
        if costos[0][0] < 1.0:  # threshold IoU (1 - IoU) < 0.5
            j = costos[0][1]
            pares.append((i, j))
            disponibles.remove(j)
    return pares


def kalman_step(x, P, Q=0.1, R=0.5):
    """Paso Kalman simple para tracking 2D: estado = [x, y, vx, vy]."""
    # x: (4,) state. P: (4, 4) covarianza.
    F = np.eye(4)
    F[0, 2] = F[1, 3] = 1.0  # position += velocity
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + np.eye(4) * Q
    # Update (mock measurement model: H = [I_2 | 0])
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    z = x[:2]  # mock: observamos la posicion
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + np.eye(2) * R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(4) - K @ H) @ P_pred
    return x_new, P_new


def main() -> int:
    # Frame 1: 2 detecciones
    dets_t1 = [[0, 0, 10, 10], [20, 20, 30, 30]]
    # Frame 2: 2 detecciones (movieron un poco)
    dets_t2 = [[1, 1, 11, 11], [22, 22, 32, 32]]
    iou = iou_matrix(dets_t1, dets_t2)
    print(f"IoU matrix:\n{iou}")
    # Asignacion
    cost = 1.0 - iou
    pares = hungarian_assignment(cost)
    print(f"Asignaciones: {pares}")
    # Kalman
    x = np.array([0.0, 0.0, 1.0, 1.0])
    P = np.eye(4)
    for _ in range(5):
        x, P = kalman_step(x, P)
    print(f"Estado Kalman tras 5 pasos: {x.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())