"""
Lección: 07-rlhf
Fase: 10
RLHF (Reinforcement Learning from Human Feedback) para LLMs.
SFT -> Reward Model -> PPO. Mas detalle en fase 09/09.
"""
from __future__ import annotations
import sys
import numpy as np


def bradley_terry_loss(chosen_reward, rejected_reward):
    return float(-np.log(1.0 / (1.0 + np.exp(-(chosen_reward - rejected_reward))) + 1e-9))


def rlhf_pipeline_steps():
    return [
        "1. Pre-training: foundation model",
        "2. SFT: instruction tuning en datos curados",
        "3. Reward Model: Bradley-Terry con preferences humanas",
        "4. PPO: policy=SFT, reference=SFT frozen, reward=RM",
        "5. KL constraint: KL(pi || pi_ref) * beta",
        "6. Iterative: round 2, 3, ... con nuevos preferences",
    ]


def reward_hacking_risks():
    return [
        "Verbose outputs: optimiza 'length' del RM",
        "Mode collapse: un solo estilo",
        "Jailbreaks: exploit RM gaps",
        "Reward gaming: tokens raros",
    ]


def rlhf_components():
    return {
        "SFT model": "Base para policy",
        "Reward model": "BT loss, scalar",
        "Reference model": "SFT frozen, KL target",
        "Policy (LoRA)": "SFT + LoRA, trainable",
        "Value head": "Critico para PPO",
        "KL penalty": "beta=0.05-0.1",
    }


def main() -> int:
    # BT loss
    loss = bradley_terry_loss(1.0, 0.0)
    print(f"BT loss: {loss:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())