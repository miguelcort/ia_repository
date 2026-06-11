"""
Lección: 25-speculative-decoding
Fase: 10
Speculative decoding (Leviathan 2023, Chen 2023): draft chico, valida con target.
Acceptance-rejection criterion, distribution preservation.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def acceptance_rejection(p_target, p_draft, draft_token, eps=1e-9):
    """Acceptance: min(1, p_target(t) / p_draft(t)).
    """
    p_t = max(p_target[draft_token], eps)
    p_d = max(p_draft[draft_token], eps)
    return min(1.0, p_t / p_d)


def adjusted_distribution(p_target, p_draft, eps=1e-9):
    """Si se rechaza, p' = max(0, p_target - p_draft), renormalize.
    """
    p_prime = np.maximum(p_target - p_draft, 0)
    s = p_prime.sum()
    if s < eps:
        return p_target  # fallback
    return p_prime / s


def expected_speedup(acceptance_rate, k_draft, draft_cost_ratio=0.1):
    """Expected speedup de spec decoding.
    Speedup = E[accepted] / (1 + k_draft * draft_cost_ratio)
    E[accepted] = sum_{i=1..k} acceptance_rate^i = (1 - r^(k+1)) / (1 - r) - 1.
    """
    if acceptance_rate >= 0.99:
        return (k_draft + 1) / (1 + k_draft * draft_cost_ratio)
    if acceptance_rate < 1e-3:
        return 1.0 / (1 + k_draft * draft_cost_ratio)
    expected = sum(acceptance_rate ** i for i in range(1, k_draft + 1))
    return expected / (1 + k_draft * draft_cost_ratio)


def draft_model_simple(prompt, draft_model):
    """Mock draft model: genera k tokens."""
    return [draft_model.sample(prompt) for _ in range(draft_model.k)]


def main() -> int:
    # Demo
    p_target = np.array([0.5, 0.3, 0.2])
    p_draft = np.array([0.4, 0.4, 0.2])
    token = 0
    a = acceptance_rejection(p_target, p_draft, token)
    print(f"Acceptance for token {token}: {a:.3f}")
    # Speedup
    sp = expected_speedup(0.7, k_draft=4)
    print(f"Expected speedup (r=0.7, k=4): {sp:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())