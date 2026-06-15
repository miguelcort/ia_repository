# 07 — End-to-end fine-tuning pipeline

> Pipeline completo: data curation (decontamination, dedup, quality filter) → SFT → DPO/IPO → eval → merge → deploy. Incluye: dataset prep, training (LoRA, full FT), eval (lm-eval-harness), quantization, serving (vLLM, TGI).

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 10 (LLMs), Fase 11 (eng), Fase 17 (infra)
**Tiempo estimado:** 30 horas

## Objetivos

- Curate dataset (decontaminate, dedup, quality).
- SFT con LoRA o full FT.
- DPO/IPO alignment.
- Eval + quantization + deploy.

## El problema

Pipeline completo de fine-tuning en producción:
(1) Data curation: download corpus (FineWeb,
RedPajama), decontaminate vs evals, dedup
(MinHash, suffix array), quality filter (heuristic
+ classifier). (2) SFT: tokenize, train LoRA
(rank 16-64) o full FT, cosine LR, bf16, gradient
checkpointing, FSDP. (3) Alignment: DPO o IPO
sobre preferences. (4) Eval: lm-eval-harness,
custom evals. (5) Quantization: GPTQ, AWQ, GGUF.
(6) Deploy: vLLM, TGI, SGLang, merged weights.

## Constrúyelo

```python
from trl import SFTTrainer, DPOTrainer
from peft import LoraConfig
from datasets import load_dataset


def full_finetune_pipeline(model_name, dataset_name, output_dir):
    # 1. Load
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)

    # 2. SFT
    lora_config = LoraConfig(r=64, lora_alpha=128)
    sft_trainer = SFTTrainer(model, dataset,
                            peft_config=lora_config)
    sft_trainer.train()

    # 3. DPO
    dpo_trainer = DPOTrainer(model, dataset["preferences"])
    dpo_trainer.train()

    # 4. Save
    model.save_pretrained(output_dir)
    return model
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-finetune-pipeline
fase: 19
leccion: 07
---

1. Data curation.
2. SFT con LoRA.
3. DPO alignment.
4. Eval (lm-eval-harness).
5. Quantization + serve.
```

## Ejercicios

1. **SFT**: fine-tune Llama 3 8B con
   LoRA en 10K examples.
2. **DPO**: alignment round.
3. **Desafío**: full pipeline
   + serve en vLLM.

## Lecturas recomendadas

- "TRL" (Hugging Face 2024)
- "PEFT" (Hugging Face 2024)
- "vLLM" (Kwon 2023)
- "FineWeb" (Hugging Face 2024)

---

> 📚 **Adaptación al español** de la lección
> "[07-end-to-end-fine-tuning-pipeline]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
