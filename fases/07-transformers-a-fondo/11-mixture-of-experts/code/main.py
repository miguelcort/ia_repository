"""
Lección: 11-mixture-of-experts
Fase: 07
MoE: routing tokens a top-k experts. Sparse activation, dense params.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def top_k_routing(x, W_gate, n_experts, top_k):
    """Router: x @ W_gate -> probs por expert. Top-k por fila.
    x: (n_tokens, d_model). W_gate: (d_model, n_experts).
    Returns: indices (n_tokens, top_k), weights (n_tokens, top_k).
    """
    logits = x @ W_gate  # (n, n_experts)
    probs = softmax(logits, axis=-1)
    # Top-k
    topk_idx = np.argpartition(-probs, top_k, axis=-1)[..., :top_k]
    topk_w = np.take_along_axis(probs, topk_idx, axis=-1)
    # Renormalize topk weights
    topk_w = topk_w / topk_w.sum(axis=-1, keepdims=True)
    return topk_idx, topk_w


def expert_ffn(x, W1, b1, W2, b2):
    """FFN de un expert: GELU(x @ W1 + b1) @ W2 + b2."""
    # GELU
    from math import erf
    h = x @ W1 + b1
    h = 0.5 * h * (1.0 + np.vectorize(erf)(h / np.sqrt(2)))
    return h @ W2 + b2


def moe_layer(x, W_gate, experts, top_k):
    """x: (n, d). experts: list of (W1, b1, W2, b2). W_gate: (d, n_experts).
    Returns: (n, d)."""
    n_experts = len(experts)
    n, d = x.shape
    idx, w = top_k_routing(x, W_gate, n_experts, top_k)  # (n, k), (n, k)
    out = np.zeros((n, d))
    # Para cada expert, acumular salida de tokens routed a el
    for e in range(n_experts):
        # Tokens que tienen este expert en su top-k
        token_mask = (idx == e).any(axis=-1)  # (n,)
        if not token_mask.any():
            continue
        # Para cada token, sumar contribution * weight
        for t in np.where(token_mask)[0]:
            pos = np.where(idx[t] == e)[0][0]
            weight = w[t, pos]
            out[t] += weight * expert_ffn(x[t:t+1], *experts[e])[0]
    return out


def load_balancing_loss(probs, idx, n_experts, top_k):
    """Auxiliary loss para balancear carga entre experts.
    f_i = fraction of tokens routed to expert i.
    p_i = mean prob assigned to expert i.
    loss = n_experts * sum(f_i * p_i).
    """
    n_tokens = probs.shape[0]
    # f_i
    counts = np.zeros(n_experts)
    for e in range(n_experts):
        counts[e] = (idx == e).sum()
    f = counts / (n_tokens * top_k)
    # p_i
    p = probs.mean(axis=0)
    return n_experts * np.dot(f, p)


def main() -> int:
    n, d, n_experts, k = 8, 16, 4, 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)) * 0.5
    W_gate = rng.standard_normal((d, n_experts)) * 0.1
    # 4 experts
    d_ff = 32
    experts = []
    for e in range(n_experts):
        W1 = rng.standard_normal((d, d_ff)) * 0.05
        b1 = np.zeros(d_ff)
        W2 = rng.standard_normal((d_ff, d)) * 0.05
        b2 = np.zeros(d)
        experts.append((W1, b1, W2, b2))
    out = moe_layer(x, W_gate, experts, top_k=k)
    print(f"Input: {x.shape}, output: {out.shape}")
    # Routing distribution
    idx, w = top_k_routing(x, W_gate, n_experts, k)
    print(f"Top-{k} indices shape: {idx.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())