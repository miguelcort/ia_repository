"""
Lección: 10-generacion-de-video
Fase: 08
Video diffusion: 3D U-Net / Video DiT. Sora, SVD, Runway, Veo, Kling.
"""
from __future__ import annotations
import sys
import numpy as np


def temporal_attention(x, n_heads, d_k, seed=0):
    """Attention sobre dimension tiempo.
    x: (T, B, H*W, D). Attention: tokens del mismo spatial location a traves de T.
    """
    rng = np.random.default_rng(seed)
    T, B, HW, D = x.shape
    # Reshape a (B*HW, T, D) para attention por spatial
    x_reshaped = x.transpose(1, 2, 0, 3).reshape(B * HW, T, D)
    return rng.standard_normal(x_reshaped.shape).reshape(B, HW, T, D).transpose(2, 0, 1, 3)


def causal_temporal_attention(x, seed=0):
    """Causal attention sobre time: cada t solo ve [0, t].
    Implementado como causal mask triangular.
    """
    T = x.shape[0]
    mask = np.triu(np.ones((T, T)) * -1e9, k=1)
    return mask  # mock


def spacetime_patches(video, patch_size=2, temporal_patch=2):
    """Divide video en patches espacio-temporales.
    video: (T, H, W, C). patches: (T/tp, H/ps, W/ps, tp*ps*ps*C).
    """
    T, H, W, C = video.shape
    T = (T // temporal_patch) * temporal_patch
    H = (H // patch_size) * patch_size
    W = (W // patch_size) * patch_size
    v = video[:T, :H, :W, :]
    patches = []
    for t in range(0, T, temporal_patch):
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                p = v[t:t+temporal_patch, i:i+patch_size, j:j+patch_size, :]
                patches.append(p.flatten())
    return np.stack(patches)


def sora_components():
    """Sora-like components."""
    return {
        "DiT backbone": "Diffusion transformer, scale to video",
        "3D VAE": "Compress video a latent (T/4, H/8, W/8, 4)",
        "Spacetime patches": "(3, 16, 16) patches, ~13M tokens para 60s 1080p",
        "Text encoder": "T5-XXL o CLIP",
        "Positional": "RoPE 3D (espacio + tiempo)",
        "Inference": "Rectified flow + CFG",
    }


def video_metrics():
    """Metricas para video generation."""
    return {
        "FVD": "Frechet Video Distance",
        "IS (video)": "Inception Score for video",
        "CLIP score": "Text-video alignment",
        "Subject consistency": "Frame-to-frame consistency",
        "Motion smoothness": "Temporal smoothness",
    }


def main() -> int:
    print("=== Sora components ===")
    for k, v in sora_components().items():
        print(f"  {k:20s} {v}")
    print("\n=== Video metrics ===")
    for k, v in video_metrics().items():
        print(f"  {k:20s} {v}")
    # Demo spacetime patches
    video = np.random.default_rng(0).standard_normal((8, 32, 32, 3))
    patches = spacetime_patches(video, patch_size=8, temporal_patch=2)
    print(f"\nVideo {video.shape} -> {patches.shape[0]} patches")
    return 0


if __name__ == "__main__":
    sys.exit(main())