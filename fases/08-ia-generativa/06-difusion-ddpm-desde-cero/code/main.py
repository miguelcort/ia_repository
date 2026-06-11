"""
Lección: 06-difusion-ddpm-desde-cero
Fase: 08
DDPM (Ho 2020): forward q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps.
Reverse: p_theta(x_{t-1}|x_t). Loss: ||eps - eps_theta(x_t, t)||^2.
"""
from __future__ import annotations
import sys
import numpy as np


def linear_beta_schedule(n_steps, beta_start=1e-4, beta_end=2e-2):
    """Schedule lineal de betas. Cosine es mas estable."""
    return np.linspace(beta_start, beta_end, n_steps)


def cosine_beta_schedule(n_steps, s=0.008):
    """Cosine schedule (Improved DDPM). Mejor calidad."""
    steps = n_steps + 1
    x = np.linspace(0, n_steps, steps)
    alphas_cumprod = np.cos(((x / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def compute_alpha_bar(betas):
    """alpha_t = 1 - beta_t. alpha_bar_t = prod(alpha_s, s=1..t)."""
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alphas, alpha_bar


def q_sample(x0, t, alpha_bar, noise=None, seed=0):
    """Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps.
    t: int o array de ints. x0: shape (..., C)."""
    if noise is None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(x0.shape)
    a = alpha_bar[t]
    while a.ndim < x0.ndim:
        a = a.reshape(-1, 1)
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * noise, noise


def p_mean_variance(eps_pred, x_t, t, alpha_bar, beta):
    """Predict mean y variance de p(x_{t-1}|x_t)."""
    a_t = alpha_bar[t]
    if a_t.ndim < x_t.ndim:
        a_t = a_t.reshape(-1, 1)
    b_t = beta[t]
    if b_t.ndim < x_t.ndim:
        b_t = b_t.reshape(-1, 1)
    # Mean: mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta)
    alpha_t = 1.0 - b_t
    coef = b_t / np.sqrt(1 - a_t)
    mean = (x_t - coef * eps_pred) / np.sqrt(alpha_t)
    # Variance: beta_t (fixed)
    var = b_t
    return mean, var


def ddpm_loss(eps_pred, eps_true):
    """MSE entre ruido predicho y real."""
    return float(((eps_pred - eps_true) ** 2).mean())


def sample_loop(eps_pred_fn, x_T, betas, alpha_bar, n_steps):
    """Reverse process: T -> 0.
    eps_pred_fn: x_t, t -> eps_pred.
    """
    x_t = x_T
    for t in reversed(range(n_steps)):
        eps = eps_pred_fn(x_t, t)
        mean, var = p_mean_variance(eps, x_t, t, alpha_bar, betas)
        if t > 0:
            rng = np.random.default_rng(t)
            noise = rng.standard_normal(x_t.shape)
        else:
            noise = 0
        x_t = mean + np.sqrt(var) * noise
    return x_t


def main() -> int:
    n_steps = 100
    betas = linear_beta_schedule(n_steps)
    alphas, alpha_bar = compute_alpha_bar(betas)
    print(f"alpha_bar[0]={alpha_bar[0]:.3f}, alpha_bar[-1]={alpha_bar[-1]:.5f}")
    # Sample
    x0 = np.random.default_rng(0).standard_normal((2, 4))
    t = 50
    x_t, noise = q_sample(x0, t, alpha_bar, seed=0)
    print(f"x0: {x0.shape}, x_t (t={t}): {x_t.shape}")
    print(f"x0 mean={x0.mean():.3f}, x_t mean={x_t.mean():.3f}, std={x_t.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())