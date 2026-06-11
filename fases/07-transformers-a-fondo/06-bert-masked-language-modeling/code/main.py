"""
Lección: 06-bert-masked-language-modeling
Fase: 07
BERT: MLM (masked language modeling) + NSP (next sentence prediction).
"""
from __future__ import annotations
import sys
import numpy as np


def mask_tokens(token_ids, mask_id, vocab_size, mlm_prob=0.15, seed=0):
    """15% de tokens son mask candidates. De esos:
    - 80% -> [MASK]
    - 10% -> random token
    - 10% -> unchanged
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    labels = np.full(len(token_ids), -100)  # -100 = ignore en loss
    # Indices candidatos
    prob_matrix = rng.uniform(size=len(token_ids)) < mlm_prob
    # No maskear [CLS], [SEP] (IDs 0, 1)
    special = np.isin(token_ids, [0, 1])
    prob_matrix &= ~special
    indices = np.where(prob_matrix)[0]
    if len(indices) == 0:
        return token_ids.copy(), labels
    labels[indices] = token_ids[indices]
    # 80% MASK
    mask_indices = rng.uniform(size=len(indices)) < 0.8
    input_ids = list(token_ids)
    for i, m in zip(indices, mask_indices):
        if m:
            input_ids[i] = mask_id
    # 10% random
    rest = indices[~mask_indices]
    if len(rest) > 0:
        random_idx = rng.uniform(size=len(rest)) < 0.5  # 10% de 15% = 50% de 50%
        # De los que NO fueron MASK (20%), mitad son random, mitad unchanged
        # ya hicimos 80% mask, quedan 20% -> 10% random, 10% unchanged
        for i, r in zip(rest, random_idx):
            if r:
                input_ids[i] = int(rng.integers(0, vocab_size))
    return np.array(input_ids), labels


def compute_mlm_loss(logits, labels):
    """Cross-entropy solo en posiciones con label != -100."""
    if (labels == -100).all():
        return 0.0
    mask = labels != -100
    # logits: (n, vocab), labels: (n,)
    p = np.exp(logits[mask] - logits[mask].max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    nll = -np.log(p[np.arange(len(labels[mask])), labels[mask]])
    return float(nll.mean())


def create_nsp_labels(sent_a, sent_b, is_next):
    """NSP: 1 si sent_b sigue a sent_a, 0 si no."""
    return int(is_next), [sent_a, sent_b]


def main() -> int:
    # Demo
    tokens = [101, 2024, 1996, 3006, 102, 3231, 1997, 102]  # [CLS] the cat sat [SEP] on a [SEP]
    masked, labels = mask_tokens(tokens, mask_id=103, vocab_size=30000, seed=0)
    print("Original: ", tokens)
    print("Masked:   ", masked.tolist())
    print("Labels:   ", labels.tolist())
    # Loss demo
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((len(tokens), 30522))
    loss = compute_mlm_loss(logits, labels)
    print(f"MLM loss (random logits): {loss:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())