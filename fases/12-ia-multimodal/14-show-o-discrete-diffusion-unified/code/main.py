"""
Lección: 14-show-o-discrete-diffusion-unified
Fase: 12
Show-o (2024): discrete diffusion unified. Single transformer.
Text + image tokens en un solo vocab. Masked token prediction.
"""
from __future__ import annotations
import numpy as np


def show_o_vocab(vqvae_codebook=8192, text_vocab=50000, special=128):
    """Vocab unificado."""
    return vqvae_codebook + text_vocab + special


def add_mask(tokens, mask_id, p_mask=0.5, seed=0):
    """Mask tokens con probabilidad p_mask."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(tokens)) < p_mask
    out = tokens.copy()
    out[mask] = mask_id
    return out, mask


def show_o_denoise_step(masked_tokens, predicted_logits, mask_id, t):
    """Replace masked positions con predicted tokens."""
    out = masked_tokens.copy()
    # argmax over logits
    pred_ids = predicted_logits.argmax(axis=-1)
    # reemplazar masked positions (en out los masked son mask_id)
    mask_positions = out == mask_id
    out[mask_positions] = pred_ids[mask_positions]
    return out


def show_o_loss(predicted_logits, target_tokens, mask_positions):
    """Cross-entropy solo sobre masked positions."""
    if not mask_positions.any():
        return 0.0
    e = np.exp(predicted_logits - predicted_logits.max(axis=-1, keepdims=True))
    log_probs = np.log(e / e.sum(axis=-1, keepdims=True))
    nll = -log_probs[mask_positions, target_tokens[mask_positions]]
    return float(nll.mean())


def show_o_sample_image(text_ids, n_image_tokens=64, mask_id=8192+50000+1,
                        vqvae=8192, n_steps=10):
    """Sample image via iterative denoising."""
    tokens = np.full(n_image_tokens, mask_id, dtype=np.int64)
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        # mock: random logits
        logits = rng.standard_normal((n_image_tokens, show_o_vocab())) * 0.1
        tokens = show_o_denoise_step(tokens, logits, mask_id, t=n_steps - step - 1)
    return tokens


def main() -> int:
    print(f"Show-o vocab: {show_o_vocab()}")
    tokens = np.array([10, 20, 30, 40, 50])
    masked, mask = add_mask(tokens, mask_id=99999, p_mask=0.4, seed=0)
    print(f"Masked: {masked}, mask: {mask}")
    img = show_o_sample_image(text_ids=np.array([1, 2]), n_image_tokens=16)
    print(f"Show-o image tokens: {img.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())