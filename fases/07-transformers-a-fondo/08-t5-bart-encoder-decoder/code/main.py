"""
Lección: 08-t5-bart-encoder-decoder
Fase: 07
T5: text-to-text transfer. BART: denoising autoencoder. Encoder-decoder.
"""
from __future__ import annotations
import sys
import numpy as np


def span_corrupt(token_ids, mask_id, span_length=3, ratio=0.15, seed=0):
    """T5-style: reemplaza spans consecutivos con [MASK].
    Returns: input_ids, target_ids (solo en mask positions).
    """
    rng = np.random.default_rng(seed)
    n = len(token_ids)
    n_mask = max(1, int(n * ratio))
    # Sortear n_mask indices de inicio
    starts = sorted(rng.choice(n, size=min(n_mask, n), replace=False))
    targets = np.full(n, -100)  # -100 = ignore
    input_ids = list(token_ids)
    for s in starts:
        # Span de hasta span_length tokens consecutivos
        e = min(n, s + int(rng.integers(1, span_length + 1)))
        targets[s:e] = token_ids[s:e]
        for i in range(s, e):
            input_ids[i] = mask_id
    return np.array(input_ids), targets


def shift_right(input_ids, decoder_start_id=0):
    """Para T5/BART decoder: prepend decoder_start, drop last."""
    shifted = [decoder_start_id] + list(input_ids[:-1])
    return np.array(shifted)


def label_smoothing_loss(logits, targets, eps=0.1, ignore_index=-100):
    """Cross-entropy con label smoothing."""
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(len(targets)), targets]
    # Label smoothing: distribucion uniforme
    smooth = -log_probs.mean(axis=-1)  # entropy term
    loss = (1 - eps) * nll + eps * smooth
    return float(loss[mask].mean())


def bart_noise(tokens, mask_id, p_mask=0.15, p_delete=0.05, seed=0):
    """BART denoising: token masking, deletion, infilling, sentence permutation."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    out = list(tokens)
    for i in range(n):
        r = rng.uniform()
        if r < p_mask:
            out[i] = mask_id
        elif r < p_mask + p_delete:
            out[i] = -1  # delete marker
    return [t for t in out if t != -1]


def main() -> int:
    tokens = list(range(20))
    # T5
    inp, tgt = span_corrupt(tokens, mask_id=99, seed=0)
    print(f"T5 span corrupt: input {inp[:10].tolist()}...")
    # BART
    noisy = bart_noise(tokens, mask_id=99, seed=0)
    print(f"BART noisy: {noisy[:10]} (orig {len(tokens)}, noisy {len(noisy)})")
    # Shift right
    inp = np.array([5, 10, 15, 20])
    shifted = shift_right(inp, decoder_start_id=0)
    print(f"Shift right: {shifted.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())