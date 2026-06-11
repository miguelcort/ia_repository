"""
Lección: 08-dpo
Fase: 10
DPO (Direct Preference Optimization): no critic, no reward model.
Closed-form: pi(y|x) = pi_ref * exp(r/beta) / Z. Llama 3, Mistral usan DPO.
"""
from __future__ import annotations
import sys
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """DPO loss: -log sigmoid(beta * (logp_chosen - logp_rejected) - beta * (ref_chosen - ref_rejected)).
    """
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    margin = beta * (chosen_diff - rejected_diff)
    return float(-np.log(_sigmoid(margin) + 1e-9))


def dpo_loss_vectorized(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Vectorized DPO loss para batch."""
    chosen_diff = chosen_logps - ref_chosen_logps
    rejected_diff = rejected_logps - ref_rejected_logps
    margins = beta * (chosen_diff - rejected_diff)
    losses = -np.log(_sigmoid(margins) + 1e-9)
    return float(np.mean(losses))


def ipo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1, tau=0.1):
    """IPO loss: regularized DPO.
    L = (log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l)) - 1/(2*tau))^2.
    """
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    return float((chosen_diff - rejected_diff - 1.0 / (2 * tau)) ** 2)


def kto_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """KTO loss: Kahneman-Tversky Optimization, binary.
    """
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    # Use sigmoid of KL
    kl_chosen = 1.0 / (1.0 + np.exp(-chosen_diff))
    kl_rejected = 1.0 / (1.0 + np.exp(-rejected_diff))
    # KTO loss: rewards the desirable, penalizes the undesirable
    return float(beta * (kl_rejected - kl_chosen))


def dpo_components():
    return {
        "Reference model": "SFT frozen, KL target",
        "Policy model": "Trainable, SFT init",
        "Beta": "Temperature, controls KL strength",
        "Loss": "Binary cross-entropy on margin",
        "Data": "Pairs (chosen, rejected) preferences",
    }


def dpo_vs_ppo():
    return {
        "DPO": "No critic, no RM. Closed-form. Simple, +stable. Sample-inefficient",
        "PPO": "Critic + KL. RL framework. Powerful, +complex. Sample-inefficient",
        "KTO": "Binary feedback. Prospect theory. +flexible",
        "IPO": "Regularized DPO. Less overfit",
    }


def main() -> int:
    # DPO demo
    loss = dpo_loss(chosen_logp=0.0, rejected_logp=-2.0,
                    ref_chosen_logp=-1.0, ref_rejected_logp=-1.0)
    print(f"DPO loss: {loss:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())