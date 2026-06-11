"""
Lección: 07-difusion-latente-stable-diffusion
Fase: 08
Latent diffusion: VAE encoder a latente 4x downsample, diffusion en latent.
Stable Diffusion: SD 1.5, 2.1, XL, 3 (DiT), CFG.
"""
from __future__ import annotations
import sys
import numpy as np


def vae_encode(x, W_enc):
    """VAE encoder simple: downsample 4x (mock: average pool 2x2)."""
    # Asumimos x shape (H, W, C). Aplicar avg pool 4x4 -> H/4, W/4, C
    H, W, C = x.shape
    if H % 4 != 0 or W % 4 != 0:
        x = x[:H - H % 4, :W - W % 4, :]
        H, W, C = x.shape
    # Pool 4x4
    pooled = x.reshape(H // 4, 4, W // 4, 4, C).mean(axis=(1, 3))
    return pooled @ W_enc


def vae_decode(z, W_dec):
    """VAE decoder: upsample 4x."""
    sh, sw, C = z.shape
    # 4x upsample replicando
    up = np.repeat(np.repeat(z, 4, axis=0), 4, axis=1)
    return up @ W_dec


def latent_diffusion_forward(latent, t, alpha_bar, eps_pred):
    """Diffusion en latent space: z_t = sqrt(a) * z_0 + sqrt(1-a) * eps.
    t: timestep. eps_pred: ruido predicho por U-Net.
    """
    a = alpha_bar[t]
    if a.ndim < latent.ndim:
        a = a.reshape(1, 1, 1)
    return np.sqrt(a) * latent + np.sqrt(1 - a) * eps_pred


def classifier_free_guidance(eps_uncond, eps_cond, w):
    """CFG: eps_cfg = eps_uncond + w * (eps_cond - eps_uncond).
    w=1: solo condicional. w>1: amplifica condicion. w<1: menos adherencia al prompt.
    """
    return eps_uncond + w * (eps_cond - eps_uncond)


def text_conditioning(text_emb, W_proj):
    """Proyectar CLIP text embedding a cross-attn dim."""
    return text_emb @ W_proj


def sd_components():
    """Stable Diffusion components y dimensions."""
    return {
        "VAE encoder": "8x downsample, latent (4, 64, 64) para 512x512",
        "VAE decoder": "8x upsample, latent -> image",
        "U-Net": "Cross-attn layers con text emb",
        "Text encoder": "CLIP ViT-L/14 (SD1.5) o OpenCLIP (SDXL)",
        "Time embedding": "Sinusoidal, fed to U-Net",
        "CFG scale": "1-15, default 7.5",
    }


def main() -> int:
    components = sd_components()
    for k, v in components.items():
        print(f"  {k:20s} {v}")
    # Demo
    latent = np.random.default_rng(0).standard_normal((64, 64, 4))
    print(f"\nLatent shape: {latent.shape} (8x downsample from 512x512)")
    text_emb = np.random.default_rng(1).standard_normal((1, 77, 768))
    W_proj = np.random.default_rng(2).standard_normal((768, 1024)) * 0.01
    proj = text_conditioning(text_emb, W_proj)
    print(f"Text emb: {text_emb.shape} -> {proj.shape} (77 tokens, 1024 dim)")
    # CFG
    eps_uncond = np.random.default_rng(3).standard_normal((64, 64, 4))
    eps_cond = np.random.default_rng(4).standard_normal((64, 64, 4))
    eps_cfg = classifier_free_guidance(eps_uncond, eps_cond, w=7.5)
    print(f"CFG: eps shape preserved: {eps_cfg.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())