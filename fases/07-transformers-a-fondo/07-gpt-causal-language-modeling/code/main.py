"""
Lección: 07-gpt-causal-language-modeling
Fase: 07
GPT: decoder-only, causal LM, next-token prediction.
"""
from __future__ import annotations
import sys
import numpy as np


def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets):
    """logits: (seq, vocab). targets: (seq,)."""
    p = softmax(logits, axis=-1)
    nll = -np.log(p[np.arange(len(targets)), targets])
    return float(nll.mean())


def perplexity(logits, targets):
    """exp(mean cross-entropy)."""
    return float(np.exp(cross_entropy_loss(logits, targets)))


def top_k_filter(logits, k):
    """Mask logits que no estan en top-k -> -inf."""
    filtered = logits.copy()
    threshold = np.partition(filtered, -k, axis=-1)[..., -k:].min(axis=-1, keepdims=True)
    filtered[filtered < threshold] = -1e9
    return filtered


def top_p_filter(logits, p):
    """Nucleus: smallest set con prob acumulada >= p."""
    sorted_idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    probs = softmax(sorted_logits, axis=-1)
    cum = np.cumsum(probs, axis=-1)
    # Mantener hasta que cum >= p
    keep = cum <= p
    keep[..., 0] = True  # al menos el primero
    # Convertir mask a indices originales
    mask = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(mask, sorted_idx, keep, axis=-1)
    filtered = np.where(mask, logits, -1e9)
    return filtered


def sample_next_token(logits, temperature=1.0, k=0, p=1.0, seed=0):
    """Sample con temperature, top-k, top-p."""
    logits = logits / temperature
    if k > 0:
        logits = top_k_filter(logits, k)
    if p < 1.0:
        logits = top_p_filter(logits, p)
    probs = softmax(logits, axis=-1)
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(probs), p=probs / probs.sum()))


def main() -> int:
    rng = np.random.default_rng(0)
    vocab = 100
    seq = 5
    targets = rng.integers(0, vocab, size=seq)
    logits = rng.standard_normal((seq, vocab))
    ppl = perplexity(logits, targets)
    print(f"Perplexity (random logits): {ppl:.1f}")
    # Greedy
    next_id = sample_next_token(logits[-1], temperature=0.01, seed=0)
    print(f"Greedy next token: {next_id}")
    # Sampling
    next_id = sample_next_token(logits[-1], temperature=1.0, k=5, p=0.9, seed=0)
    print(f"Sampled (T=1, k=5, p=0.9): {next_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())