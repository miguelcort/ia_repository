"""
Lección: 23-difusion-transformers-y-flujo-rectificado
Fase: 04
Prerrequisitos: 11-stable-diffusion
"""
from __future__ import annotations
import sys
import numpy as np


def linear_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Schedule DDPM lineal."""
    return np.linspace(beta_start, beta_end, timesteps)


def rectificado_flow_schedule(timesteps):
    """Rectified flow: trayectoria lineal entre data y ruido.
    x_t = (1-t) * x_0 + t * epsilon, con t en [0, 1]."""
    return np.linspace(0, 1, timesteps)


def interpolacion_rectified(x0, ruido, t):
    """x_t = (1-t) * x_0 + t * ruido."""
    return (1 - t) * x0 + t * ruido


def velocidad_target(x0, ruido):
    """Target velocity para rectified flow: dx_t/dt = ruido - x0."""
    return ruido - x0


def loss_rectified(v_pred, v_target):
    """MSE entre velocidad predicha y target."""
    return float(np.mean((v_pred - v_target) ** 2))


def loss_ddpm(eps_pred, eps_real):
    """Loss DDPM clasica: MSE entre ruido predicho y real."""
    return float(np.mean((eps_pred - eps_real) ** 2))


def comparar_trayectorias(x0, ruido, n_steps=20):
    """Compara trayectorias DDPM y rectified flow."""
    ddpm_betas = linear_schedule(n_steps)
    rf_times = rectificado_flow_schedule(n_steps)
    tray_ddpm = [x0]
    tray_rf = [x0]
    x_ddpm = x0.copy()
    x_rf = x0.copy()
    # DDPM
    alphas = 1.0 - ddpm_betas
    alpha_bars = np.cumprod(alphas)
    for t in range(1, n_steps):
        ab_t = alpha_bars[t]
        x_ddpm = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * ruido
        tray_ddpm.append(x_ddpm.copy())
    # Rectified
    for t in range(1, n_steps):
        x_rf = interpolacion_rectified(x0, ruido, rf_times[t])
        tray_rf.append(x_rf.copy())
    return tray_ddpm, tray_rf


def main() -> int:
    x0 = np.array([1.0, 2.0])
    ruido = np.array([0.5, -0.3])
    v = velocidad_target(x0, ruido)
    print(f"Velocidad target: {v}")
    # Rectified flow
    v_pred = v + np.random.default_rng(0).normal(0, 0.1, size=2)
    print(f"Loss rectified: {loss_rectified(v_pred, v):.4f}")
    # Comparar trayectorias
    _, tray_rf = comparar_trayectorias(x0, ruido, n_steps=10)
    print(f"Rectified x_t en t=0.5: {interpolacion_rectified(x0, ruido, 0.5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())