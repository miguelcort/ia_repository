"""
Lección: 12-generacion-3d
Fase: 08
3D generation: NeRF, point clouds, mesh, Gaussian Splatting, 3D diffusion.
"""
from __future__ import annotations
import sys
import numpy as np


def rays_o_d(camera, image_size, focal, c2w):
    """Genera origen y direccion de cada rayo pixel.
    camera: (3,) origen. image_size: (H, W). focal: scalar. c2w: (4, 4) camera-to-world.
    Returns: rays_o (H, W, 3), rays_d (H, W, 3).
    """
    H, W = image_size
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    # Direcciones en camera space
    dirs = np.stack([(i - W * 0.5) / focal,
                     -(j - H * 0.5) / focal,
                     -np.ones_like(i)], axis=-1)
    # Transform a world space
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = np.broadcast_to(camera, rays_d.shape)
    return rays_o, rays_d


def volumetric_render(rgb, sigma, t_vals):
    """Volumetric rendering con composite alpha.
    rgb: (N, samples, 3). sigma: (N, samples, 1). t_vals: (samples,).
    Returns: (N, 3) color final.
    """
    # Alpha = 1 - exp(-sigma * delta_t)
    delta = np.concatenate([t_vals[1:] - t_vals[:-1], np.array([1e10])])
    alpha = 1.0 - np.exp(-sigma.squeeze() * delta)
    T = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    weights = alpha * T  # (N, samples)
    return (weights[..., None] * rgb).sum(axis=-2)


def gaussians_init(n_gaussians, init_scale=0.1, seed=0):
    """Inicializa 3D Gaussians para splatting.
    Cada gaussian: position (3), scale (3), rotation (4 quaternion), color (3), opacity (1).
    """
    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((n_gaussians, 3)) * init_scale
    scales = np.full((n_gaussians, 3), init_scale)
    rotations = np.zeros((n_gaussians, 4))
    rotations[:, 0] = 1.0  # quat identidad
    colors = rng.uniform(0, 1, (n_gaussians, 3))
    opacities = np.full(n_gaussians, 0.5)
    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


def projection_3d_to_2d(positions, K, c2w):
    """Project 3D points to 2D image plane.
    positions: (N, 3). K: (3, 3) intrinsics. c2w: (4, 4).
    Returns: uv (N, 2), depth (N,).
    """
    # World -> camera
    ones = np.ones((positions.shape[0], 1))
    p_h = np.concatenate([positions, ones], axis=-1)
    p_c = p_h @ c2w.T  # (N, 4)
    p_c = p_c[:, :3]
    # Project
    p_img = p_c @ K.T
    z = p_img[:, 2:3]
    uv = p_img[:, :2] / (z + 1e-9)
    return uv, z.squeeze()


def nerf_components():
    """NeRF: Neural Radiance Fields."""
    return {
        "MLP": "8 capas, 256 hidden, ReLU, skip connection en layer 4",
        "Input": "Posicion (3) + direccion (3) (peor caso)",
        "Pos encoding": "Frecuencias, 10 para pos, 4 para dir",
        "Output": "RGB (3) + density (1)",
        "Render": "Volumetric integration, 64 samples + hierarchical",
        "Training": "MSE entre render y ground truth image",
    }


def gs_components():
    """Gaussian Splatting."""
    return {
        "Primitives": "3D Gaussians, n millones",
        "Differentiable rasterizer": "Project + alpha blending, similar a NeRF pero rasterization",
        "Training": "Densificacion (clone + split), pruning",
        "Output": "Real-time 30+ fps novel view",
        "Edit": "Edit individual gaussians (delete, move)",
    }


def main() -> int:
    print("=== NeRF ===")
    for k, v in nerf_components().items():
        print(f"  {k:15s} {v}")
    print("\n=== Gaussian Splatting ===")
    for k, v in gs_components().items():
        print(f"  {k:20s} {v}")
    # Demo
    n = 100
    gauss = gaussians_init(n, seed=0)
    print(f"\n{n} Gaussians: {gauss['positions'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())