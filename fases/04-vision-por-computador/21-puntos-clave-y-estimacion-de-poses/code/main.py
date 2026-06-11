"""
Lección: 21-puntos-clave-y-estimacion-de-poses
Fase: 04
Prerrequisitos: 06-deteccion-de-objetos-yolo
"""
from __future__ import annotations
import sys
import numpy as np


def detectar_keypoints_mock(H=64, W=64, n_keypoints=17, semilla=0):
    """Mock: genera n keypoints aleatorios en la imagen."""
    rng = np.random.default_rng(semilla)
    # Simula distribucion anatomica: y en parte superior, x centrado
    keypoints = []
    for i in range(n_keypoints):
        x = float(rng.uniform(10, W - 10))
        y = float(rng.uniform(10, H - 10))
        keypoints.append((x, y))
    return keypoints


def conexiones_entre_keypoints(keypoints, conexiones):
    """Valida que los pares de conexiones tengan keypoints validos.
    Devuelve lista de tuplas (kp_a, kp_b) si existen."""
    pares = []
    for a, b in conexiones:
        if a < len(keypoints) and b < len(keypoints):
            pares.append((keypoints[a], keypoints[b]))
    return pares


def distancia_entre_keypoints(kp1, kp2):
    """Distancia euclidiana entre dos keypoints."""
    return float(np.sqrt((kp1[0] - kp2[0]) ** 2 + (kp1[1] - kp2[1]) ** 2))


def angulo_articulacion(a, b, c):
    """Angulo en b formado por a-b-c (en grados)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9))
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def pck_score(pred_kps, gt_kps, threshold=10.0):
    """Percentage of Correct Keypoints: % de predichas dentro de threshold del GT."""
    if len(pred_kps) != len(gt_kps):
        return 0.0
    correct = 0
    for p, g in zip(pred_kps, gt_kps):
        d = distancia_entre_keypoints(p, g)
        if d <= threshold:
            correct += 1
    return correct / len(gt_kps)


# Conexiones tipicas del esqueleto COCO (17 keypoints)
COCO_ESQUELETO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # cabeza
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # brazos
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # piernas
]


def main() -> int:
    kps = detectar_keypoints_mock(64, 64, n_keypoints=17, semilla=42)
    print(f"Keypoints: {len(kps)}")
    pares = conexiones_entre_keypoints(kps, COCO_ESQUELETO)
    print(f"Conexiones validas: {len(pares)}")
    # Angulo del codo (5-7-9)
    a, b, c = kps[5], kps[7], kps[9]
    ang = angulo_articulacion(a, b, c)
    print(f"Angulo del codo: {ang:.1f} grados")
    return 0


if __name__ == "__main__":
    sys.exit(main())