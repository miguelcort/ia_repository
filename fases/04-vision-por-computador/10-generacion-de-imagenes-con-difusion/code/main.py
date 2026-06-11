"""
Lección: 10-generacion-de-imagenes-con-difusion
Fase: 04
Prerrequisitos: 09-generacion-de-imagenes-gan
"""
from __future__ import annotations
import sys
import numpy as np


def beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Schedule lineal: beta_t aumenta linealmente de beta_start a beta_end."""
    return np.linspace(beta_start, beta_end, timesteps)


def precompute_diffusion(betas):
    """Calcula alpha, alpha_bar (producto cumulativo) y coeficientes utiles."""
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return alphas, alpha_bars


def forward_diffusion(x0, t, alpha_bars, semilla=0):
    """q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I).
    Devuelve x_t y el ruido epsilon."""
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0, 1, size=x0.shape)
    sqrt_ab = np.sqrt(alpha_bars[t])
    sqrt_one_minus = np.sqrt(1.0 - alpha_bars[t])
    x_t = sqrt_ab * x0 + sqrt_one_minus * ruido
    return x_t, ruido


def ddim_step(x_t, ruido_pred, t, alpha_bars_prev, alpha_bars_t):
    """Paso DDIM (Denoising Diffusion Implicit Models) simplificado.
    x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * ruido_pred."""
    # Estimar x0 desde x_t y ruido predicho
    x0_pred = (x_t - np.sqrt(1 - alpha_bars_t) * ruido_pred) / np.sqrt(alpha_bars_t)
    # Denoised
    x_prev = np.sqrt(alpha_bars_prev) * x0_pred + np.sqrt(1 - alpha_bars_prev) * ruido_pred
    return x_prev


def ruido_a_ruido(x0, alpha_bars, n_steps=20, semilla=0):
    """Genera una imagen a partir de ruido puro iterativamente (mock).
    En la practica, esto lo hace una U-Net."""
    rng = np.random.default_rng(semilla)
    x = rng.normal(0, 1, size=x0.shape)
    for t in reversed(range(1, n_steps)):
        # Mock: ruido predicho aleatorio (sustituir por U-Net)
        ruido_pred = rng.normal(0, 0.1, size=x.shape)
        alpha_bars_prev = alpha_bars[t - 1] if t > 0 else 1.0
        x = ddim_step(x, ruido_pred, t, alpha_bars_prev, alpha_bars[t])
    return x


def main() -> int:
    betas = beta_schedule(1000)
    print(f"Beta shape: {betas.shape}, range: [{betas[0]:.4f}, {betas[-1]:.4f}]")
    alphas, alpha_bars = precompute_diffusion(betas)
    print(f"alpha_bars[0]: {alpha_bars[0]:.4f}, alpha_bars[-1]: {alpha_bars[-1]:.6f}")
    # Forward diffusion en diferentes pasos
    x0 = np.ones((4, 4))
    for t in [0, 100, 500, 999]:
        x_t, ruido = forward_diffusion(x0, t, alpha_bars, semilla=42)
        print(f"t={t}: x_t mean={x_t.mean():.3f}, std={x_t.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())