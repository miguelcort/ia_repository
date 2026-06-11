"""
Lección: 06-instruction-tuning-sft
Fase: 10
Supervised fine-tuning (SFT) en instrucciones. Alpaca, Dolly, ShareGPT, Open-Hermes.
"""
from __future__ import annotations
import sys
import numpy as np
import re


def format_prompt(instruction, response=None, template="alpaca"):
    """Format prompt con template.
    Templates: alpaca, vicuna, chatml, llama-chat.
    """
    if template == "alpaca":
        s = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        if response is not None:
            s += response
        return s
    elif template == "chatml":
        s = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        if response is not None:
            s += response + "<|im_end|>"
        return s
    elif template == "llama-chat":
        s = f"<s>[INST] {instruction} [/INST]"
        if response is not None:
            s += f" {response} </s>"
        return s
    return instruction


def mask_labels(prompt_only, prompt_with_response):
    """Mask labels para loss solo en response (no en prompt).
    """
    # Asumimos prompt_with_response empieza con prompt_only
    assert prompt_with_response.startswith(prompt_only)
    labels = [-100] * len(prompt_only) + list(prompt_with_response[len(prompt_only):])
    return labels


def pack_dataset(items, max_len=2048):
    """Pack sequences de longitud variable en sequences de max_len."""
    sequences = []
    current = ""
    for item in items:
        if len(current) + len(item) > max_len:
            sequences.append(current)
            current = item
        else:
            current += item
    if current:
        sequences.append(current)
    return sequences


def decontaminate(dataset, eval_prompts):
    """Remove samples del dataset que contengan eval prompts (anti-contamination)."""
    filtered = []
    for sample in dataset:
        contains_eval = any(p in sample for p in eval_prompts)
        if not contains_eval:
            filtered.append(sample)
    return filtered


def sft_hyperparameters():
    """Hiperparametros tipicos SFT."""
    return {
        "lr": "2e-5 a 5e-5 (mas bajo que pre-training)",
        "batch_size": "128 (4-128 effective)",
        "epochs": "2-4",
        "context_len": "2048-4096",
        "warmup": "0.03 a 0.1",
        "schedule": "cosine decay",
        "weight_decay": "0.0 (SFT no WD)",
        "precision": "bf16",
    }


def datasets_summary():
    """SFT datasets principales."""
    return {
        "Alpaca": "Stanford, 52K self-instruct",
        "Dolly": "Databricks, 15K human-written",
        "ShareGPT": "60K ChatGPT conversations",
        "Open-Hermes": "Teknium, 1M+ examples",
        "UltraChat": "1.5M conversations",
        "WizardLM": "Evol-Instruct, 196K",
        "Tulu": "AllenAI, 939K mixed",
        "Magpie": "Self-instruct from Llama 3, 1M+",
    }


def main() -> int:
    # Demo
    prompt = format_prompt("Cual es la capital de Francia?", template="alpaca")
    response = format_prompt("Cual es la capital de Francia?", "Paris", template="alpaca")
    print(f"Alpaca prompt:\n{prompt[:50]}...")
    print(f"\nChatML:")
    print(format_prompt("Hola", template="chatml")[:60])
    return 0


if __name__ == "__main__":
    sys.exit(main())