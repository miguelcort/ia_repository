"""
Lección: 04-flamingo-gated-cross-attention
Fase: 12
Flamingo (DeepMind 2022): few-shot multimodal in-context learning.
Gated cross-attention layers inserted entre LLM blocks. tanh gate.
"""
from __future__ import annotations
import sys
import numpy as np


def gated_cross_attention(text_h, vision_kv, W_q, W_kv, W_o, gate_logit_zero_init=True):
    """text_h: (n_text, d_text). vision_kv: (n_vis, d_vis).
    Returns (n_text, d_text)."""
    d = text_h.shape[-1]
    q = text_h @ W_q
    if vision_kv.shape[-1] == d:
        k = vision_kv @ W_kv
        v = vision_kv @ W_kv
    else:
        # project vision features from d_vis to d using a small projection
        rng = np.random.default_rng(hash(vision_kv.shape) & 0xFFFFFFFF)
        W_vis = rng.standard_normal((vision_kv.shape[-1], d)) * 0.1
        vision_proj = vision_kv @ W_vis
        k = vision_proj @ W_kv
        v = vision_proj @ W_kv
    scores = q @ k.T / np.sqrt(d)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)
    out = weights @ v
    out = out @ W_o
    gate = 0.0 if gate_logit_zero_init else 1.0
    return text_h + np.tanh(gate) * out


def flamingo_block(text_h, vision_kv, W_q, W_kv, W_o, W_ffn1, W_ffn2):
    """Block: gated cross-attn + FFN (residual)."""
    h = gated_cross_attention(text_h, vision_kv, W_q, W_kv, W_o)
    ffn = np.maximum(h @ W_ffn1, 0) @ W_ffn2
    return h + ffn


def flamingo_forward(text_features, vision_features_list, n_layers=2, d=512):
    """text_features: (n, d). vision_features_list: list of (n_v, d)."""
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((d, d)) * 0.1
    Wkv = rng.standard_normal((d, d)) * 0.1
    Wo = rng.standard_normal((d, d)) * 0.1
    Wf1 = rng.standard_normal((d, d * 4)) * 0.1
    Wf2 = rng.standard_normal((d * 4, d)) * 0.1
    h = text_features
    for layer in range(n_layers):
        # alterna cross-attn con cada vision set
        vis = vision_features_list[layer % len(vision_features_list)]
        h = flamingo_block(h, vis, Wq, Wkv, Wo, Wf1, Wf2)
    return h


def main() -> int:
    rng = np.random.default_rng(0)
    text = rng.standard_normal((10, 512)) * 0.1
    vis1 = rng.standard_normal((64, 512)) * 0.1
    vis2 = rng.standard_normal((49, 512)) * 0.1
    out = flamingo_forward(text, [vis1, vis2], n_layers=2, d=512)
    print(f"Flamingo output: {out.shape}")
    print(f"Gate=0 al inicio -> cross-attn bloqueada")
    return 0


if __name__ == "__main__":
    sys.exit(main())