"""
Lección: 08-ppo
Fase: 09
PPO (Proximal Policy Optimization, Schulman 2017): clipped objective,
on-policy, simple, estable. SOTA para LLMs (RLHF, GRPO).
"""
from __future__ import annotations
import sys
import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_advantages(rewards, values, dones, gamma, lambda_):
    """GAE advantage."""
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lambda_ * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages


def ppo_clipped_objective(ratio, advantage, clip=0.2):
    """L_CLIP = min(r * A, clip(r, 1-eps, 1+eps) * A).
    r = pi_new / pi_old.
    """
    clipped_ratio = np.clip(ratio, 1 - clip, 1 + clip)
    return np.minimum(ratio * advantage, clipped_ratio * advantage)


def kl_penalty(pi_old, pi_new, eps=1e-9):
    """KL(pi_old || pi_new) aproximada. KL(old || new).
    """
    ratio = pi_new / (pi_old + eps)
    return float(np.sum(pi_old * np.log(ratio + eps)))


def ppo_clip_loss(advantages, old_log_probs, new_log_probs, clip=0.2, vf_coef=0.5, returns=None, values=None):
    """PPO loss con policy loss + value loss + entropy bonus.
    L = -L_CLIP + vf_coef * (V - R)^2 - entropy_coef * H.
    Aqui sin entropy.
    """
    ratio = np.exp(new_log_probs - old_log_probs)
    pg_loss = -np.mean(ppo_clipped_objective(ratio, advantages, clip))
    if returns is not None and values is not None:
        vf_loss = np.mean((values - returns) ** 2)
    else:
        vf_loss = 0.0
    return float(pg_loss + vf_coef * vf_loss)


def ppo_components():
    """PPO components."""
    return {
        "Clipped objective": "min(r*A, clip(r, 1-eps, 1+eps)*A), eps=0.2",
        "Advantage": "GAE, lambda=0.95 tipico",
        "Value loss": "MSE entre V(s) y returns",
        "Entropy bonus": "H(pi) para exploration",
        "Multiple epochs": "K=3-10 epochs per batch, clip asegura estabilidad",
    }


def ppo_algorithm():
    """Pseudocodigo PPO."""
    return {
        "1. Collect trajectories": "pi_old interactua con env, genera (s, a, r, v, logp)",
        "2. Compute advantages": "GAE: A_t = sum (gamma*lambda)^i * delta",
        "3. Update for K epochs": "theta = argmin L_CLIP + vf_coef*L_VF - ent_coef*H",
        "4. Mini-batches": "Split trajectories en mini-batches, shuffle",
        "5. Clip ratio": "min(r*A, clip(r, 1-eps, 1+eps)*A)",
    }


def main() -> int:
    advantages = np.array([1.0, -0.5, 0.0, 2.0])
    old_log_probs = np.zeros(4)
    new_log_probs = np.array([0.1, -0.1, 0.0, 0.3])
    loss = ppo_clip_loss(advantages, old_log_probs, new_log_probs, clip=0.2)
    print(f"PPO loss: {loss:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())