"""
Lección: 07-open-weight-vlm-recipes
Fase: 12
Open weight VLM recipes: LLaVA, Idefics, OpenFlamingo, Idefics2, Molmo.
Frameworks: transformers, vLLM, SGLang. Tipos de recipes: pretraining,
finetuning, instruction tuning, RLHF.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


VLM_RECIPES = {
    "llava-1.5": {
        "name": "LLaVA-1.5",
        "vision": "CLIP-ViT-L-336px",
        "llm": "Vicuna-13B",
        "projection": "MLP (2 layers)",
        "training": "Pretrain projector -> Visual instruction tuning",
        "params": 13_000_000_000,
        "context": 4096,
    },
    "llava-next": {
        "name": "LLaVA-Next",
        "vision": "CLIP-ViT-L-336px (any resolution, 1-4 tiles)",
        "llm": "Nous-Hermes-2-Yi-34B",
        "projection": "MLP",
        "training": "Visual instruction tuning with any-res",
        "params": 34_000_000_000,
        "context": 4096,
    },
    "idefics2": {
        "name": "Idefics2",
        "vision": "SigLIP (any res)",
        "llm": "Mistral-7B",
        "projection": "Perceiver Resampler",
        "training": "Multimodal pretrain + instruct",
        "params": 8_000_000_000,
        "context": 4096,
    },
    "open-flamingo": {
        "name": "OpenFlamingo",
        "vision": "CLIP ViT-L/14",
        "llm": "MPT / RedPajama",
        "projection": "Perceiver Resampler",
        "training": "Cross-attn in LLM layers (Flamingo-style)",
        "params": 9_000_000_000,
        "context": 2048,
    },
    "molmo": {
        "name": "Molmo",
        "vision": "CLIP ViT-L (any res, point ref)",
        "llm": "OLMo-7B",
        "projection": "Connector + point ref",
        "training": "Molmo dataset (PixMo)",
        "params": 7_000_000_000,
        "context": 4096,
    },
}


def list_recipes():
    """List all open VLM recipes."""
    return list(VLM_RECIPES.keys())


def get_recipe(name):
    """Get recipe details."""
    return VLM_RECIPES.get(name)


def recipes_by_param_count(min_b=7, max_b=70):
    """Filter recipes by parameter count (billions)."""
    result = []
    for r in VLM_RECIPES.values():
        b = r["params"] / 1e9
        if min_b <= b <= max_b:
            result.append(r)
    return result


def training_stage_to_data(stage):
    """Map training stage to datasets."""
    mapping = {
        "pretrain_projector": ["LCS-558K (image captions)"],
        "instruction_tuning": ["LLaVA-1.5 instruct 558K", "GPT-4V generated"],
        "rlhf": ["LLaVA-RLHF 10K"],
        "any_res_instruct": ["LLaVA-Next 1.4M"],
    }
    return mapping.get(stage, [])


def main() -> int:
    print("Open VLM recipes:")
    for name in list_recipes():
        r = get_recipe(name)
        print(f"  {r['name']}: {r['vision']} + {r['llm']} ({r['params']/1e9:.1f}B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())