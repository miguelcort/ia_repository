"""
Lección: 04-pre-training-mini-gpt
Fase: 10
Mini-GPT: pre-training pipeline end-to-end. Tokenize corpus, train GPT-2 small.
Karpathy's nanoGPT.
"""
from __future__ import annotations
import sys
import numpy as np
import math


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)


def cross_entropy(logits, targets):
    """logits: (seq, vocab). targets: (seq,)."""
    p = _softmax(logits, axis=-1)
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.log(p[np.arange(len(targets)), targets]).mean())


def perplexity(logits, targets):
    return float(np.exp(cross_entropy(logits, targets)))


def lr_schedule(step, warmup_steps, max_steps, max_lr, min_lr=0.1 * 1e-3):
    """Cosine decay con warmup (Karpathy)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coeff


def adamw_step(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
    """AdamW update step."""
    beta1, beta2 = betas
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param)
    return param, m, v


def grad_clip(grad, max_norm=1.0):
    """Clip gradient norm."""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


def batch_end_to_end(logits, targets):
    """Compute loss y accuracy en un batch."""
    loss = cross_entropy(logits, targets)
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == targets).mean())
    return loss, acc


def pretraining_config():
    """Config tipica de pre-training GPT-2 small."""
    return {
        "model": "GPT-2 small (124M)",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "context_len": 1024,
        "vocab_size": 50257,
        "batch_size": 0.5e6,  # tokens
        "tokens": 10e9,
        "lr_max": 6e-4,
        "weight_decay": 0.1,
        "warmup": 2000,
        "cosine_decay": True,
        "precision": "bf16",
        "GPUs": "32x A100",
        "time": "1-2 weeks",
    }


def main() -> int:
    # Demo
    np.random.default_rng(0)
    n_steps = 100
    max_lr = 6e-4
    for step in [0, 50, 200, 500, 1000]:
        lr = lr_schedule(step, warmup_steps=200, max_steps=n_steps, max_lr=max_lr)
        print(f"Step {step}: lr = {lr:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())