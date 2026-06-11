"""
Lección: 22-gaussian-splatting-3d
Fase: 04
Prerrequisitos: 13-vision-3d-nerf
"""
from __future__ import annotations
import sys
import numpy as np


def inicializar_gaussiana(pos, escala=0.1, color=(1.0, 0.0, 0.0), opacidad=1.0):
    """Inicializa una gaussiana 3D con posicion, escala, color, opacidad.
    Representacion: (pos[3], escala[3], color[3], opacidad[1], rot[4])."""
    return {
        "pos": np.array(pos, dtype=np.float32),
        "escala": np.array([escala, escala, escala], dtype=np.float32),
        "color": np.array(color, dtype=np.float32),
        "opacidad": float(opacidad),
        "rot": np.array([1.0, 0.0, 0.0, 0.0]),  # quaternion identidad
    }


def proyectar_gaussiana_2d(g, matriz_proyeccion):
    """Proyecta gaussiana 3D a elipsoide 2D. Aprox simplificada."""
    pos_h = np.append(g["pos"], 1.0)
    p_2d = matriz_proyeccion @ pos_h
    if p_2d[3] == 0:
        return None
    p_2d = p_2d[:3] / p_2d[3]
    return {"centro": p_2d[:2], "color": g["color"], "opacidad": g["opacidad"]}


def composicion_gaussianas(g_2d_list, H, W):
    """Alpha compositing: front-to-back blending."""
    # Inicializa imagen vacia (RGBA)
    color = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    for g in g_2d_list:
        cx, cy = g["centro"]
        cx, cy = int(cx), int(cy)
        if 0 <= cx < W and 0 <= cy < H:
            alpha_new = alpha[cy, cx] + g["opacidad"] * (1 - alpha[cy, cx])
            if alpha_new > 0:
                color[cy, cx] = (color[cy, cx] * alpha[cy, cx] + g["color"] * g["opacidad"]) / alpha_new
            alpha[cy, cx] = alpha_new
    return color, alpha


def densidad_gaussiana(g, punto):
    """Densidad de probabilidad en 'punto' segun la gaussiana (esférica)."""
    diff = punto - g["pos"]
    dist2 = float((diff ** 2).sum())
    sigma2 = float((g["escala"] ** 2).mean())
    return float(np.exp(-dist2 / (2 * sigma2)))


def main() -> int:
    # 3 gaussianas para un cubo simple
    gs = [
        inicializar_gaussiana([0, 0, 5], color=(1, 0, 0)),
        inicializar_gaussiana([1, 0, 5], color=(0, 1, 0)),
        inicializar_gaussiana([0, 1, 5], color=(0, 0, 1)),
    ]
    # Proyeccion simple
    P = np.array([[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    gs_2d = [proyectar_gaussiana_2d(g, P) for g in gs]
    print(f"Gaussianas 2D: {[(g['centro'], g['color']) for g in gs_2d if g]}")
    # Densidad
    for g in gs:
        d = densidad_gaussiana(g, np.array([0, 0, 5]))
        print(f"Densidad en centro: {d:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())