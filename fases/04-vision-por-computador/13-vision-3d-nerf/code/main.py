"""
Lección: 13-vision-3d-nerf
Fase: 04
Prerrequisitos: 04-clasificacion-de-imagenes
"""
from __future__ import annotations
import sys
import numpy as np


def coordenadas_3d_a_2d(puntos_3d, fx=500, fy=500, cx=320, cy=240):
    """Proyecta puntos 3D (X, Y, Z) a 2D (u, v) con pinhole camera model."""
    X, Y, Z = puntos_3d[..., 0], puntos_3d[..., 1], puntos_3d[..., 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.stack([u, v], axis=-1)


def rayo_direccion(u, v, fx=500, fy=500, cx=320, cy=240):
    """Direccion de un rayo a traves del pixel (u, v)."""
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.stack([x / norm, y / norm, z / norm], axis=-1)


def muestrear_rayos(origen, direccion, t_near=1.0, t_far=10.0, n_muestras=64):
    """Muestrea puntos a lo largo de un rayo entre t_near y t_far."""
    t = np.linspace(t_near, t_far, n_muestras)
    puntos = origen + direccion[..., None, :] * t[..., None]
    return puntos, t


def nerf_densidad_color(puntos, semilla=0):
    """Mock de NeRF: devuelve densidad y color para cada punto 3D."""
    rng = np.random.default_rng(semilla)
    densidad = np.exp(-np.linalg.norm(puntos, axis=-1) ** 2 / 4.0)
    color = 0.5 + 0.5 * rng.normal(0, 1, size=puntos.shape)
    color = np.tanh(color[..., :3])
    return densidad, color


def composicion_volumen(densidad, color, t, delta):
    """Composicion de alpha a lo largo del rayo: C = sum(T_i * alpha_i * c_i)."""
    sigma = densidad
    alpha = 1.0 - np.exp(-sigma * delta)
    T = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    pesos = T * alpha
    color_final = (pesos[..., None] * color).sum(axis=-2)
    profundidad = (pesos * t).sum(axis=-1)
    return color_final, profundidad, pesos


def main() -> int:
    # Genera un rayo
    dir = rayo_direccion(np.array([320.0]), np.array([240.0]))
    print(f"Ray dir shape: {dir.shape}")
    origen = np.zeros(3)
    puntos, t = muestrear_rayos(origen, dir, t_near=1.0, t_far=5.0, n_muestras=32)
    print(f"Puntos shape: {puntos.shape}")
    sigma, color = nerf_densidad_color(puntos)
    delta = float(t[1] - t[0])
    color_final, profundidad, pesos = composicion_volumen(sigma, color, t, delta)
    print(f"Color final: {color_final}, profundidad: {profundidad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())