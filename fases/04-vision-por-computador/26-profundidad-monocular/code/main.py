"""
Lección: 26-profundidad-monocular
Fase: 04
Prerrequisitos: 04-clasificacion-de-imagenes
"""
from __future__ import annotations
import sys
import numpy as np


def depth_to_3d(depth, K, xy):
    """Convierte pixel (u, v) + depth a punto 3D usando intrinsics K."""
    u, v = xy
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth[int(v), int(u)]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def depth_evaluation(pred, gt, threshold=1.25):
    """Metrica delta < threshold: proporcion de pixeles donde
    max(pred/gt, gt/pred) < threshold.
    """
    ratio = np.maximum(pred / (gt + 1e-9), gt / (pred + 1e-9))
    return float((ratio < threshold).mean())


def depth_rmse(pred, gt):
    """RMSE entre profundidad predicha y ground truth."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def depth_mae(pred, gt):
    """MAE en metros."""
    return float(np.mean(np.abs(pred - gt)))


def absolute_relative_error(pred, gt):
    """AbsRel: |pred - gt| / gt promediado."""
    return float(np.mean(np.abs(pred - gt) / (gt + 1e-9)))


def main() -> int:
    # Mock: depth map y ground truth
    H, W = 32, 32
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 10.0, size=(H, W))
    gt = depth + rng.normal(0, 0.1, size=(H, W))
    K = np.array([[500.0, 0, 16], [0, 500.0, 16], [0, 0, 1.0]])
    # Pixel (16, 16) -> 3D
    p3d = depth_to_3d(depth, K, (16, 16))
    print(f"Punto 3D (centro): {p3d}")
    # Metricas
    delta = depth_evaluation(depth, gt, threshold=1.25)
    rmse = depth_rmse(depth, gt)
    mae = depth_mae(depth, gt)
    rel = absolute_relative_error(depth, gt)
    print(f"delta<1.25: {delta:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, AbsRel: {rel:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())