"""
Lección: 11-chameleon-early-fusion-tokens
Fase: 12
Chameleon (Meta 2024): early fusion. Image tokens discretizados a
VQ-VAE codes. Text + image tokens en mismo vocabulario.
"""
from __future__ import annotations
import numpy as np


def vqvae_encode(image, codebook_size=8192, patch_size=16):
    """VQ-VAE: image -> discrete codes (n_patches,)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    n = (H // patch_size) * (W // patch_size)
    return rng.integers(0, codebook_size, size=n)


def vqvae_decode(codes, codebook, target_size):
    """Discrete codes -> reconstructed image. (n,) -> (H, W, C)."""
    patch_size = 16
    H, W, C = target_size
    nh, nw = H // patch_size, W // patch_size
    reconstructed = codebook[codes]
    # reconstructed: (nh*nw, patch_size^2 * C)
    return reconstructed.reshape(nh, nw, patch_size, patch_size, C).transpose(0, 2, 1, 3, 4).reshape(H, W, C)


def chameleon_vocab(vocab_size_text=32000, vocab_size_image=8192, special_tokens=128):
    """Vocabulario unificado. Total = text + image + special."""
    return vocab_size_text + vocab_size_image + special_tokens


def encode_text(text_ids, vocab_size_text):
    """Text ids: < vocab_size_text."""
    return np.array([i if i < vocab_size_text else -1 for i in text_ids])


def encode_image_tokens(image_codes, vocab_size_text, special_offset=64):
    """Image codes -> vocab ids (offset)."""
    return image_codes + vocab_size_text + special_offset


def chameleon_sequence(text_ids, image_codes, vocab_size_text):
    """Concat text + image tokens en secuencia unica."""
    text = encode_text(text_ids, vocab_size_text)
    img = encode_image_tokens(image_codes, vocab_size_text)
    return np.concatenate([text, img])


def main() -> int:
    img = np.random.default_rng(0).standard_normal((256, 256, 3))
    codes = vqvae_encode(img, codebook_size=8192, patch_size=16)
    print(f"Image tokens: {codes.shape} discrete codes")
    text_ids = np.array([10, 20, 30, 40])
    seq = chameleon_sequence(text_ids, codes, vocab_size_text=32000)
    print(f"Chameleon sequence: {seq.shape}")
    print(f"Total vocab: {chameleon_vocab()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())