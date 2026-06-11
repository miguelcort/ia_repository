"""
Lección: 15-janus-pro-decoupled-encoders
Fase: 12
Janus-Pro (DeepSeek 2025): decoupled encoders. Encoder separado
para understanding vs generation. SigLIP encoder para understanding,
VQ tokenizer para generation. Single transformer unified.
"""
from __future__ import annotations
import numpy as np


def janus_understanding_encoder(image, embed_dim=1152):
    """SigLIP-style encoder para understanding. image -> (n_tokens, embed_dim)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    patch_size = 14
    n = (H // patch_size) * (W // patch_size) + 1
    return rng.standard_normal((n, embed_dim)) * 0.1


def janus_generation_encoder(image, vqvae=32768, patch_size=16):
    """VQ tokenizer para generation. image -> (n_patches,) discrete codes."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    n = (H // patch_size) * (W // patch_size)
    return rng.integers(0, vqvae, size=n)


def janus_unified_embed(image_features_und, image_features_gen, text_ids,
                        embed_dim=4096, und_offset=0, gen_offset=2048, text_offset=4096):
    """Unified embedding: image understanding + image generation + text."""
    # los 3 sources mapean al mismo embed_dim
    und_W = np.eye(image_features_und.shape[-1], embed_dim)
    gen_W = np.eye(embed_dim, embed_dim)
    text_W = np.eye(text_ids.shape[-1] if text_ids.ndim == 2 else 1, embed_dim)
    return None  # placeholder


def janus_route_modality(encoder_type, modality):
    """Router: understanding encoder para image-text, generation encoder para gen."""
    routing = {
        "understanding": "siglip",
        "generation": "vqvae",
    }
    return routing[encoder_type]


def janus_forward(image, text_ids, mode="understanding", embed_dim=4096):
    """image + text -> LLM dim.
    mode: 'understanding' o 'generation'."""
    if mode == "understanding":
        img_feats = janus_understanding_encoder(image, embed_dim=embed_dim)
    else:
        img_codes = janus_generation_encoder(image, vqvae=32768)
        # convert codes a embed_dim (mock)
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((32768, embed_dim)) * 0.02
        img_feats = emb[img_codes]
    # concat with text (mock text -> embed_dim)
    text_emb = np.zeros((text_ids.shape[0], embed_dim))
    return np.concatenate([img_feats, text_emb], axis=0)


def main() -> int:
    img = np.random.default_rng(0).standard_normal((224, 224, 3))
    und_feats = janus_understanding_encoder(img)
    print(f"Janus-Pro understanding: {und_feats.shape}")
    gen_codes = janus_generation_encoder(img)
    print(f"Janus-Pro generation codes: {gen_codes.shape}")
    out = janus_forward(img, np.array([1, 2, 3]), mode="understanding")
    print(f"Janus-Pro forward: {out.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())