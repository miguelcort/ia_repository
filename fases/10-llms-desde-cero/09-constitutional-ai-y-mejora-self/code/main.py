"""
Lección: 09-constitutional-ai-y-mejora-self
Fase: 10
Constitutional AI (Anthropic 2022): reglas + AI feedback en lugar de human.
RLAIF (AI feedback), self-critique, revision.
"""
from __future__ import annotations
import sys
import numpy as np


def constitutional_principles():
    """Principios comunes de Constitutional AI."""
    return [
        "No harmful content",
        "No illegal activity",
        "No personal data exposure",
        "Be helpful, harmless, honest",
        "Respect human autonomy",
        "Be transparent about AI nature",
    ]


def rlaif_steps():
    """Pasos de Constitutional AI / RLAIF."""
    return [
        "1. Sample response from model",
        "2. Self-critique against constitution",
        "3. AI generates revision based on critique",
        "4. Train on (prompt, original, critique, revision)",
        "5. RLAIF: train RM con AI preferences (no human)",
        "6. Iterative rounds",
    ]


def self_critique_score(response, principles):
    """Mock self-critique: cuenta cuantos principles violados."""
    # Mock: 30% chance violate each
    rng = np.random.default_rng(0)
    violations = sum(int(rng.uniform() < 0.3) for _ in principles)
    return float(violations) / len(principles)


def ai_preference_alignment(chosen, rejected):
    """Mock AI preference: chosen deberia ser mas aligned."""
    return chosen


def constitutional_vs_rlhf():
    return {
        "RLHF": "Human preferences, RM + PPO. Costoso: human labeling",
        "RLAIF": "AI feedback, no humans. Mas barato, escalable",
        "Constitutional AI": "Reglas + AI critique. RLAIF + revision",
        "Self-improvement": "Iteration sin humans. AI self-critique",
    }


def main() -> int:
    print("=== Constitutional AI principles ===")
    for p in constitutional_principles():
        print(f"  - {p}")
    print("\n=== RLAIF steps ===")
    for s in rlaif_steps():
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())