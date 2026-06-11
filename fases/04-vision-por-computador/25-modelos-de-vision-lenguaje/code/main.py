"""
Lección: 25-modelos-de-vision-lenguaje
Fase: 04
Prerrequisitos: 18-clip-vocabulario-abierto
"""
from __future__ import annotations
import sys
import numpy as np


def image_to_tokens(img, patch_size=14, dim=1024, semilla=0):
    """Divide imagen en patches y proyecta a dim tokens."""
    H, W, C = img.shape
    n_patches = (H // patch_size) * (W // patch_size)
    rng = np.random.default_rng(semilla)
    W_proj = rng.normal(scale=np.sqrt(1.0 / (patch_size * patch_size * C)), size=(patch_size * patch_size * C, dim))
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size, :].flatten()
            patches.append(patch @ W_proj)
    return np.array(patches)


def text_to_tokens(text, n_tokens, dim=1024, semilla=0):
    """Mock: convierte texto en n_tokens embeddings."""
    rng = np.random.default_rng(hash(text) % 2**32)
    return rng.normal(0, 1, size=(n_tokens, dim))


def llava_compose(image_tokens, text_tokens):
    """LLaVA concatena image tokens + text tokens. Input al LLM."""
    return np.vstack([image_tokens, text_tokens])


def greedy_decode(logits, tokenizer_vocab, max_new=10):
    """Mock greedy decoding."""
    generated = []
    for _ in range(max_new):
        idx = int(np.argmax(logits))
        if idx < len(tokenizer_vocab):
            generated.append(tokenizer_vocab[idx])
        if generated and generated[-1] == "<eos>":
            break
        # Mock: nuevo logits = logits + aleatorio
        logits = logits + np.random.default_rng(len(generated)).normal(0, 0.1, size=logits.shape)
    return generated


def main() -> int:
    img = np.random.default_rng(0).normal(size=(224, 224, 3))
    image_tokens = image_to_tokens(img, patch_size=14, dim=1024)
    print(f"Image tokens: {image_tokens.shape}")
    text_tokens = text_to_tokens("Describe this image:", n_tokens=8, dim=1024)
    print(f"Text tokens: {text_tokens.shape}")
    composed = llava_compose(image_tokens, text_tokens)
    print(f"Composed: {composed.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())