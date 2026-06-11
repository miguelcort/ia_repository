"""
Lección: 13-flow-matching-y-rectified-flows
Fase: 08
Flow matching (Lipman 2022): ODE path entre distribuciones. Rectified flow: straight paths.
SD3, Stable Cascade, Flux usan flow matching.
"""
from __future__ import annotations
import sys
import numpy as np


def linear_interp(x0, x1, t):
    """Interpolacion lineal entre x0 (t=0) y x1 (t=1)."""
    return (1 - t) * x0 + t * x1


def conditional_flow(x0, x1, t):
    """Conditional flow: x_t = t * x1 + (1 - t) * x0.
    Velocity: dx_t/dt = x1 - x0.
    """
    return linear_interp(x0, x1, t), x1 - x0


def flow_matching_loss(v_pred, x1, x0, t):
    """Loss: ||v_pred - (x1 - x0)||^2."""
    target = x1 - x0
    return float(((v_pred - target) ** 2).mean())


def sample_flow(model_fn, x0, n_steps, dim):
    """ODE solver: x_{t+dt} = x_t + v_theta(x_t, t) * dt.
    Euler method."""
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = i * dt
        v = model_fn(x, t)
        x = x + v * dt
    return x


def rectified_flow_straightness(x0, x1, v_pred_fn, n_steps=10):
    """Rectified flow: minimizar curvature del path.
    Straight paths <=> menos ODE steps, rapido.
    """
    dt = 1.0 / n_steps
    path = [x0]
    x = x0
    for i in range(n_steps):
        v = v_pred_fn(x, i * dt)
        x = x + v * dt
        path.append(x)
    return np.array(path)


def flow_matching_vs_diffusion():
    """Comparacion."""
    return {
        "Diffusion (DDPM)": "SDE, 1000 steps, noise -> data via score matching",
        "Flow matching (FM)": "ODE, 10-50 steps, conditional flow learning",
        "Rectified flow (RF)": "Linear path, optimal transport, straight",
        "SD3 uses": "Rectified flow + MM-DiT",
        "FLUX uses": "Flow matching + MM-DiT",
        "Stability AI": "Adopted RF en SD3, mejor que DDPM",
    }


def sde_to_ode_concept():
    """Concepto: SDE tiene noise, ODE deterministico."""
    return {
        "SDE": "dx = f(x, t) dt + g(t) dW. Stochastic.",
        "ODE (probability flow)": "dx = f(x, t) - 0.5 * g(t)^2 * score(x, t) dt",
        "Flow matching": "Aprende direct velocity, sin noise.",
        "Rectified flow": "Special case, straight line between distributions.",
    }


def main() -> int:
    print("=== Flow matching vs diffusion ===")
    for k, v in flow_matching_vs_diffusion().items():
        print(f"  {k:20s} {v}")
    # Demo
    x0 = np.zeros((2, 4))
    x1 = np.ones((2, 4))
    t = 0.5
    x_t, v = conditional_flow(x0, x1, t)
    print(f"\nAt t={t}: x_t mean={x_t.mean():.3f}, velocity={v[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())