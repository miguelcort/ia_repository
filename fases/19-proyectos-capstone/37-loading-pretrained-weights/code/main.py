"""
Lección: 37-loading-pretrained-weights
Fase: 19
Capstone de ingeniería AI: 37 Loading Pretrained Weights.
"""
from __future__ import annotations
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pretrained(model_id, dtype=torch.bfloat16,
                  quantization="none"):
    kwargs = {"torch_dtype": dtype}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
