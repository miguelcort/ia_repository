"""
Lección: 14-construye-un-transformer-capstone
Fase: 07
Capstone: transformer decoder completo para text generation.
Stack: tokenizer mock -> embeddings -> N bloques -> LM head.
"""
from __future__ import annotations
import sys
import numpy as np
import math


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))


def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)


def attn(q, k, v, mask=None):
    d_k = q.shape[-1]
    s = q @ k.T / np.sqrt(d_k)
    if mask is not None:
        s = s + mask
    w = softmax(s, axis=-1)
    return w @ v


class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = s * rng.standard_normal((d_model, d_model))
        self.W_K = s * rng.standard_normal((d_model, d_model))
        self.W_V = s * rng.standard_normal((d_model, d_model))
        self.W_O = s * rng.standard_normal((d_model, d_model))
        self.W1 = s * rng.standard_normal((d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = s * rng.standard_normal((d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x, mask):
        a = attn(x @ self.W_Q, x @ self.W_K, x @ self.W_V, mask=mask)
        h = layer_norm(x + a @ self.W_O)
        return layer_norm(h + gelu(h @ self.W1 + self.b1) @ self.W2 + self.b2)


class TransformerLM:
    """Decoder-only transformer para text generation."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq=64, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.token_emb = s * rng.standard_normal((vocab_size, d_model))
        # Positional: sinusoidal
        self.pos_emb = self._sinusoidal(max_seq, d_model)
        self.blocks = [DecoderBlock(d_model, n_heads, d_ff, seed=seed + i + 1)
                       for i in range(n_layers)]
        self.W_head = s * rng.standard_normal((vocab_size, d_model))
        self.d_model = d_model

    def _sinusoidal(self, seq, d):
        pos = np.arange(seq).reshape(-1, 1)
        i = np.arange(d).reshape(1, -1)
        angle = pos / (10000 ** (2 * (i // 2) / d))
        pe = np.zeros((seq, d))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        return pe

    def forward(self, token_ids):
        """token_ids: (seq,). Returns: logits (seq, vocab)."""
        seq = len(token_ids)
        x = self.token_emb[token_ids]
        x = x + self.pos_emb[:seq]
        mask = causal_mask(seq)
        for block in self.blocks:
            x = block(x, mask)
        return x @ self.W_head.T

    def generate(self, prompt_ids, max_new=5, temperature=1.0, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        for _ in range(max_new):
            logits = self.forward(np.array(ids))[-1] / temperature
            probs = softmax(logits)
            next_id = int(rng.choice(len(probs), p=probs / probs.sum()))
            ids.append(next_id)
        return ids


def main() -> int:
    vocab_size = 100
    model = TransformerLM(vocab_size, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq=32)
    prompt = [5, 10, 15]
    out = model.generate(prompt, max_new=5, temperature=1.0, seed=0)
    print(f"Prompt: {prompt}")
    print(f"Generated: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())