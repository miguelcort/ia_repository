# 39 — Instruction tuning (SFT)

> SFT (supervised fine-tuning): entrenar LM en pares (prompt, response) con cross-entropy. Datasets: Alpaca, Dolly, OpenHermes, Tulu, UltraChat. Frameworks: TRL, axolotl, LLaMA-Factory.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/37
**Tiempo estimado:** ~25 minutos

## Objetivos

- Load SFT dataset.
- Format con chat template.
- Train con SFTTrainer.
- Eval.

## Constrúyelo

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def sft_train(model_id, dataset_id, output_dir):
    model, tokenizer = load_pretrained(model_id)
    dataset = load_dataset(dataset_id)
    config = SFTConfig(output_dir=output_dir,
                       num_train_epochs=3,
                       per_device_train_batch_size=2,
                       gradient_accumulation_steps=8)
    trainer = SFTTrainer(model, config, tokenizer, dataset)
    trainer.train()
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
fase: 19
leccion: 39
---

1. SFT dataset.
2. Chat template.
3. SFTTrainer.
4. Eval (MMLU, IFeval).
```

## Ejercicios

1. **Alpaca**: 52K examples
   SFT Llama 3 8B.
2. **Tulu 3**: 1M examples.
3. **Desafío**: custom
   domain SFT.

## Detalles

SFT datasets: Alpaca (52K, self-instruct),
Dolly (15K, human), OpenHermes 2.5 (1M, mixed),
Tulu 3 (1M+ curated, AI2), UltraChat (1.5M, multi-
turn), ShareGPT (90K, real conversations), LIMA (1K,
high quality, "less is more").

Chat template: estructura de mensajes. Llama 3:
`<|begin_of_text|><|start_header_id|>user<|end_header_id|>

...<|eot_id|><|start_header_id|>assistant<|end_header_id|>
...<|eot_id|>`. Mistral: `<s>[INST] ... [/INST]`. ChatML
(OpenHermes): `<|im_start|>user\n...<|im_end|>`.
Reusar tokenizer = reusar template.

TRL SFTTrainer: SFTConfig con packing, completion-only
loss, masking del prompt. Packing: concat examples
hasta window size. Completion-only: loss solo sobre
response (no prompt). Mejora quality.

Multi-turn SFT: estructura [(user_1, asst_1),
(user_2, asst_2)]. Mask user_2 también. Más complejo
pero necesario para chat.

Hoy: Tulu 3 SFT (1M examples, multi-source) + DPO
(280K pairs, UltraFeedback) es el standard.

SFT vs prompting: SFT para format, capabilities
específicas. Prompting para prototipos. SFT + DPO
para assistants.

## Lecturas recomendadas

- "Alpaca" (Taori 2023)
- "OpenHermes" (Teknium 2023)
- "Tulu 3" (AI2 2024)
- "LIMA" (Zhou 2023)
- "TRL" (Hugging Face 2024)

---

> 📚 **Adaptación al español** de la lección
> "[39-instruction-tuning-sft]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
