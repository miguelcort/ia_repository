"""
Lección: 16-speculative-decoding
Fase: 07
Speculative decoding: borrador chico, valida con modelo grande. Speedup 2-3x.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sample_token(probs, seed=0):
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(probs), p=probs / probs.sum()))


def acceptance_rejection(p_draft, p_target, token, eps=1e-9):
    """Probability de aceptar token del draft: min(1, p_target / p_draft).
    Para speculative decoding, queremos que la distribucion target sea la final.
    """
    p_d_t = max(p_draft[token], eps)
    p_t_t = max(p_target[token], eps)
    return min(1.0, p_t_t / p_d_t)


def adjust_distribution(p_draft, p_target, accepted_token, gamma):
    """Si se rechaza, ajustar distribucion: p' = max(0, p_target - p_draft), renormalizar.
    gamma: numero de tokens speculados.
    """
    # p' = max(0, p_target - p_draft)
    p_prime = np.maximum(p_target - p_draft, 0)
    p_prime = p_prime / (p_prime.sum() + 1e-9)
    return p_prime


class SpeculativeDecoder:
    """Speculative decoding con draft model (chico) y target model (grande)."""

    def __init__(self, draft_sampler, target_sampler, gamma=4):
        """draft_sampler: (context, k) -> (tokens, probs).
        target_sampler: (context, k) -> probs.
        gamma: numero de tokens a especular por step.
        """
        self.draft = draft_sampler
        self.target = target_sampler
        self.gamma = gamma

    def step(self, context, seed=0):
        """Genera hasta gamma+1 tokens (los gamma draft + 1 verified)."""
        # 1. Draft: generar gamma tokens con draft model
        draft_tokens, draft_probs = self.draft(context, self.gamma, seed=seed)
        # 2. Target: evaluar probs para todos los gamma draft tokens (en paralelo)
        target_probs = self.target(context, draft_tokens)
        # 3. Accept/reject cada token
        accepted = []
        for i, tok in enumerate(draft_tokens):
            r = acceptance_rejection(draft_probs[i], target_probs[i], tok)
            rng = np.random.default_rng(seed + i)
            if rng.uniform() < r:
                accepted.append(tok)
            else:
                # Rechazar: sample from adjusted distribution
                p_adj = adjust_distribution(draft_probs[i], target_probs[i], tok, self.gamma)
                new_tok = sample_token(p_adj, seed=seed + 100 + i)
                accepted.append(new_tok)
                break
        # 4. Si todos aceptados, anadir uno mas del target
        if len(accepted) == self.gamma:
            extra = sample_token(target_probs[-1], seed=seed + 200)
            accepted.append(extra)
        return accepted


def mock_draft_sampler(context, k, seed=0):
    """Mock: genera k tokens con distribucion 'draft' (chico)."""
    rng = np.random.default_rng(seed)
    tokens = [int(rng.integers(0, 10)) for _ in range(k)]
    # Probs: uniformes (mock)
    probs = [np.ones(10) / 10 for _ in range(k)]
    return tokens, probs


def mock_target_sampler(context, draft_tokens, seed=0):
    """Mock: target probs (grande)."""
    # Target probs similares a draft, small variation
    return [np.ones(10) / 10 for _ in draft_tokens]


def benchmark_speedup(seq_len, gamma=4):
    """Estima speedup de speculative decoding.
    Speedup ~ (gamma + 1) / (gamma + ratio_draft_target)
    ratio_draft_target = cost_draft / cost_target
    """
    # Asumimos draft 10x mas rapido
    cost_draft_per_token = 1
    cost_target_per_token = 10
    # Sin speculative: target genera 1 token por step
    cost_no_spec = seq_len * cost_target_per_token
    # Con speculative: target genera 1 token (verifica gamma) + (algo draft)
    # Amortizado: por cada gamma+1 tokens, target hace 1 forward (de gamma+1 tokens)
    # Draft hace gamma forwards de 1 token
    cost_spec = seq_len * (cost_target_per_token + gamma * cost_draft_per_token) / (gamma + 1)
    return cost_no_spec / cost_spec


def main() -> int:
    decoder = SpeculativeDecoder(mock_draft_sampler, mock_target_sampler, gamma=4)
    context = [1, 2, 3]
    out = decoder.step(context, seed=0)
    print(f"Speculative step output: {out}")
    speedup = benchmark_speedup(seq_len=100, gamma=4)
    print(f"Speedup estimado: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())