"""
Lección: 09-reward-modeling-y-rlhf
Fase: 09
Reward modeling: entrenar RM con preferences (Bradley-Terry).
RLHF: SFT -> RM -> PPO. DPO, KTO, GRPO variants.
"""
from __future__ import annotations
import sys
import numpy as np


def bradley_terry_loss(chosen_reward, rejected_reward):
    """BT loss: -log sigmoid(r_chosen - r_rejected).
    Asume recompensa escalar, chosen > rejected.
    """
    return float(-np.log(_sigmoid(chosen_reward - rejected_reward) + 1e-9))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_accuracy(chosen_rewards, rejected_rewards):
    """Fraccion de pares donde chosen > rejected."""
    return float((chosen_rewards > rejected_rewards).mean())


def rlhf_pipeline_steps():
    """Pasos del pipeline RLHF."""
    return [
        "1. SFT (Supervised Fine-Tuning) en instrucciones",
        "2. Reward Model entrenado con human preferences (Bradley-Terry)",
        "3. PPO: policy = SFT, reference = SFT (frozen), reward = RM, KL penalty",
        "4. Inference: SFT + PPO-aligned model",
    ]


def dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """DPO loss: -log sigmoid(beta * (logp_chosen - logp_rejected) - beta * (ref_chosen - ref_rejected)).
    DPO: no critic, no reward model. Direct preference optimization.
    """
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    return float(-np.log(_sigmoid(beta * (chosen_diff - rejected_diff)) + 1e-9))


def kto_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """KTO (Kahneman-Tversky Optimization) loss.
    Simplificado: similar a DPO pero con aspired/desired status.
    """
    # KTO usa prospect theory
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    # KL divergences aproximadas
    kl_chosen = chosen_diff.mean()
    kl_rejected = rejected_diff.mean()
    return float(beta * (kl_rejected - kl_chosen))


def grpo_loss(rewards, clip=0.2, eps=1e-9):
    """GRPO loss: A_i = (r_i - mean) / std.
    L = -E[min(r * A, clip * A)].
    """
    advantages = (rewards - rewards.mean()) / (rewards.std() + eps)
    ratios = np.ones_like(rewards)  # Mock: r=1
    clipped = np.clip(ratios, 1 - clip, 1 + clip)
    return float(-np.minimum(ratios * advantages, clipped * advantages).mean())


def reward_hacking_warning():
    """Por que RM puede hacer hack."""
    return "RM proxy != true human preference. Optimizing RM puede no mejorar real quality. Mitigaciones: KL penalty, iterative RLHF, ensemble RMs."


def main() -> int:
    # BT loss
    chosen, rejected = 1.0, 0.0
    loss = bradley_terry_loss(chosen, rejected)
    print(f"BT loss (chosen=1, rejected=0): {loss:.3f}")
    # Pairwise accuracy
    chosen_r = np.array([1.0, 0.5, 2.0])
    rejected_r = np.array([0.0, 0.7, 1.0])
    acc = pairwise_accuracy(chosen_r, rejected_r)
    print(f"Pairwise accuracy: {acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())