"""
Lección: 11-stable-diffusion
Fase: 04
Prerrequisitos: 10-generacion-de-imagenes-con-difusion
"""
from __future__ import annotations
import sys
import numpy as np


def simulador_text_encoder(prompt):
    """Mock: convierte prompt en embedding de dimension fija (e.g. CLIP, 768)."""
    # Hash simple a vector deterministico
    rng = np.random.default_rng(hash(prompt) % 2**32)
    return rng.normal(0, 1, size=(1, 768))


def simulador_unet(x_t, t, text_emb):
    """Mock: U-Net predice el ruido. Devuelve un ruido del mismo tamano que x_t."""
    rng = np.random.default_rng(t)
    return rng.normal(0, 1, size=x_t.shape) * 0.1


def cfg(eps_cond, eps_uncond, w=7.5):
    """Classifier-free guidance: amplifica condicion, resta uncond."""
    return eps_uncond + w * (eps_cond - eps_uncond)


def simulador_vae_decode(latent):
    """Mock: VAE decoder de latent (4, H/8, W/8) a imagen (3, H, W)."""
    # Latent shape (1, 4, 64, 64) -> (1, 3, 512, 512) idealmente
    C_lat, H_lat, W_lat = latent.shape[1], latent.shape[2], latent.shape[3]
    return np.random.default_rng(0).normal(0, 1, size=(1, 3, H_lat * 8, W_lat * 8))


def simulador_vae_encode(image):
    """Mock: VAE encoder de imagen (3, H, W) a latent (4, H/8, W/8)."""
    _, _, H, W = image.shape
    return np.random.default_rng(0).normal(0, 1, size=(1, 4, H // 8, W // 8))


def main() -> int:
    prompt = "a beautiful sunset over the mountains"
    text_emb = simulador_text_encoder(prompt)
    print(f"Text emb shape: {text_emb.shape}")
    # Mock pipeline
    x = np.random.default_rng(0).normal(0, 1, size=(1, 4, 64, 64))
    for t in [999, 750, 500, 250, 100, 50, 0]:
        eps = simulador_unet(x, t, text_emb)
        x = x - 0.1 * eps
    img = simulador_vae_decode(x)
    print(f"Imagen shape: {img.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())