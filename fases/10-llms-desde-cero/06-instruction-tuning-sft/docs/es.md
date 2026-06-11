# Instruction tuning (SFT)

> SFT: pre-trained LLM entrenado en pares (instruction, response). Datasets: Alpaca (52K), Dolly (15K), ShareGPT (60K), Open-Hermes (1M+), UltraChat (1.5M), WizardLM, Tulu (939K), Magpie (1M+), SmolTalk. Templates: Alpaca, ChatML (`<|im_start|>`/`<|im_end|>`), Llama-chat (`<s>[INST]`). Loss: solo en response (mask prompt con -100). LR bajo (2e-5 a 5e-5), 2-4 epochs, batch 128. Decontamination: 8-gram overlap contra eval. Variantes: full, LoRA, QLoRA (4-bit+LoRA, 65B en 48GB), DoRA. Frameworks: TRL, axolotl, LLaMA-Factory, unsloth.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar prompt templates (Alpaca, ChatML, Llama).
- Implementar label masking.
- Implementar sequence packing.
- Diagnosticar hiperparámetros SFT.

## Constrúyelo

```python
def format_prompt(instruction, response=None, template="alpaca"):
    if template == "alpaca":
        s = f"### Instruction:\n{instruction}\n\n### Response:\n"
        if response: s += response
        return s
    elif template == "chatml":
        s = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        if response: s += response + "<|im_end|>"
        return s
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sft
fase: 10
leccion: 06
---

1. SFT: instruction, response pairs.
2. Loss solo en response.
3. LR 2e-5 a 5e-5, 2-4 epochs.
4. Alpaca, ChatML, Llama templates.
5. QLoRA para custom domain.
```

## Ejercicios

1. **SFT**: fine-tune Llama 3 8B
   con TRL en Alpaca.
2. **QLoRA**: configurar 4-bit
   base + LoRA, 65B en 48GB.
3. **Desafio**: SFT multi-turn
   conversations con ShareGPT.

## Lecturas recomendadas

- "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (Wang et al., 2022)
- "Alpaca: A Strong, Replicable Instruction-Following Model" (Stanford, 2023)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- "TRL: Transformer Reinforcement Learning" (HuggingFace)

---

> 📚 **Adaptación al español** de la lección "[Instruction Tuning SFT]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).