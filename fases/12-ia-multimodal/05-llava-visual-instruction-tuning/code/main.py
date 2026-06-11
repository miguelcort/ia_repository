"""
Lección: 05-llava-visual-instruction-tuning
Fase: 12
LLaVA (Liu 2023): visual instruction tuning. Linear projection ViT -> LLM space.
Conversational + instruction following. 158K instances, GPT-4 generated.
"""
from __future__ import annotations
import sys
import numpy as np


def clip_vit_forward(image, patch_size=14, embed_dim=1024):
    """Simula ViT forward: image (H, W, C) -> (n_patches + 1, embed_dim)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    n = (H // patch_size) * (W // patch_size) + 1
    return rng.standard_normal((n, embed_dim)) * 0.1


def visual_projection(image_features, W_proj):
    """Linear projection: ViT dim -> LLM dim."""
    return image_features @ W_proj


def build_prompt(system_msg, image_tokens, user_msg, tokenizer_encode=None):
    """Construye prompt con image tokens. Returns list of dicts."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "<image>"},
    ]
    return messages


def llava_forward(image_features, text_input_emb, W_proj, llm_forward_fn):
    """image -> ViT -> projection -> concat text -> LLM -> output."""
    image_emb = visual_projection(image_features, W_proj)
    full_input = np.vstack([image_emb, text_input_emb])
    return llm_forward_fn(full_input)


def conversation_format(question, answer, image_placeholder="<image>"):
    """Formatea conversacion LLaVA-style."""
    return f"USER: {image_placeholder}\n{question}\nASSISTANT: {answer}"


def mask_targets(prompt_ids, response_start, ignore_index=-100):
    """Loss solo sobre tokens de respuesta."""
    labels = np.full(len(prompt_ids), ignore_index, dtype=np.int64)
    labels[response_start:] = np.array(prompt_ids[response_start:])
    return labels


def main() -> int:
    img = np.random.default_rng(0).standard_normal((224, 224, 3))
    vit_out = clip_vit_forward(img, patch_size=14, embed_dim=1024)
    print(f"ViT output: {vit_out.shape}")
    W = np.random.default_rng(0).standard_normal((1024, 4096)) * 0.02
    img_emb = visual_projection(vit_out, W)
    print(f"Image embeddings (en LLM space): {img_emb.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())