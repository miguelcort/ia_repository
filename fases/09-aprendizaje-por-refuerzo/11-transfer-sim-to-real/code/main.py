"""
Lección: 11-transfer-sim-to-real
Fase: 09
Sim-to-real: entrenar en simulacion, deploy en real.
Domain randomization, system identification, action space.
"""
from __future__ import annotations
import sys
import numpy as np


def domain_randomization_param(param_name, distribution, low, high, seed=0):
    """Samplea un param de simulacion random para DR.
    distribution: 'uniform' o 'normal'.
    """
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(low, high)
    elif distribution == "normal":
        return rng.normal((low + high) / 2, (high - low) / 4)
    return 0.0


def add_observation_noise(obs, noise_std=0.1, seed=0):
    """Anade ruido gaussiano a obs para robustez sim-to-real."""
    rng = np.random.default_rng(seed)
    return obs + rng.normal(0, noise_std, obs.shape)


def system_id_residual(real_obs, sim_obs):
    """Residuals entre real y sim. Usado para fine-tune del sim.
    """
    return real_obs - sim_obs


def dynamics_randomization():
    """Tipos de domain randomization."""
    return {
        "Visual": "Color, lighting, texture, camera angle",
        "Dynamics": "Mass, friction, damping, motor strength",
        "Sensor": "Noise, latency, dropout",
        "Action": "Delay, jitter, noise",
        "Environmental": "Wind, terrain, gravity",
    }


def reality_gap_mitigations():
    """Mitigations del reality gap."""
    return {
        "Domain randomization": "Randomize sim params, train robust",
        "System identification": "Fit sim a real data, update sim",
        "Domain adaptation": "Adapt sim features to real (CycleGAN, etc)",
        "Real-world fine-tuning": "Train en sim, fine-tune en real",
        "Sim-to-real via cycle-consistency": "CycleGAN entre sim y real",
    }


def action_space_check(actions_real, actions_sim, tolerance=0.05):
    """Verifica que las acciones del sim son compatibles con real.
    """
    return np.all(np.abs(actions_real - actions_sim) < tolerance)


def main() -> int:
    # DR
    friction = domain_randomization_param("friction", "uniform", 0.5, 1.0, seed=0)
    print(f"Friction DR sample: {friction:.3f}")
    # Obs noise
    obs = np.array([1.0, 2.0, 3.0])
    noisy = add_observation_noise(obs, noise_std=0.1, seed=0)
    print(f"Noisy obs: {noisy.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())