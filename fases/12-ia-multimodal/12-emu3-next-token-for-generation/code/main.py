"""
Lección: 12-emu3-next-token-for-generation
Fase: 12
Emu3 (BAAI 2024): next-token prediction is all you need. Unifies
understanding + generation en single token space.
"""
from __future__ import annotations
import numpy as np


def emu3_tokenizer(visual_codebook_size=32768, text_vocab_size=100352, special_tokens=128):
    """Emu3 tokenizer: visual + text + special."""
    return visual_codebook_size + text_vocab_size + special_tokens


def emu3_image_tokens(image, vqvae):
    """Image -> discrete VQ tokens via VQ-VAE."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    patch_size = 16
    n = (H // patch_size) * (W // patch_size)
    return rng.integers(0, vqvae, size=n)


def emu3_understand(image_tokens, text_ids, visual_codebook=32768, text_vocab=100352, special=128):
    """Concatenar image + text en un solo vocab.
    Image: [0, visual_codebook)
    Text: [visual_codebook, visual_codebook + text_vocab)
    Special: [visual_codebook + text_vocab, ...)."""
    img_offset = 0
    txt_offset = visual_codebook + special
    img_ids = image_tokens + img_offset
    txt_ids = text_ids + txt_offset
    return np.concatenate([img_ids, txt_ids])


def emu3_generate(prefix_ids, model_fn, n_new=64, temperature=1.0):
    """Autoregresivo: prefix -> n_new tokens via sampling."""
    ids = list(prefix_ids)
    rng = np.random.default_rng(0)
    for _ in range(n_new):
        logits = model_fn(np.array(ids))
        # sample
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        next_id = rng.choice(len(probs), p=probs)
        ids.append(int(next_id))
    return np.array(ids)


def main() -> int:
    img = np.random.default_rng(0).standard_normal((256, 256, 3))
    img_tokens = emu3_image_tokens(img, vqvae=32768)
    text_ids = np.array([100, 200, 300])
    seq = emu3_understand(img_tokens, text_ids)
    print(f"Emu3 sequence: {seq.shape}")
    # mock model: uniform logits
    def mock_model(x):
        return np.ones(emu3_tokenizer()) / emu3_tokenizer()
    gen = emu3_generate(np.array([1, 2, 3]), mock_model, n_new=16)
    print(f"Emu3 generated: {gen.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())