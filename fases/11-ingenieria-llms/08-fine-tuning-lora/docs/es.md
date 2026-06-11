# Fine-tuning LoRA

> LoRA (Hu 2021, Microsoft): low-rank adaptation. W_new = W + (α/r)·A·B. A in (in, r), B in (r, out), A random init, B=0. W frozen, solo A, B entrenables. Reducción 100-1000x de trainable params. Merge: W_new = W + (α/r)·A·B, no overhead en inference. Variantes SOTA 2024-25: QLoRA (Dettmers 2023, NF4 4-bit + LoRA + double quant, 65B en 48GB), DoRA (Liu 2024, decomposed magnitude + direction, +quality), rsLoRA (rank-stabilized scaling), AdaLoRA (adaptive rank), LongLoRA (long context, shift+re-norm), LoRA+. Frameworks: PEFT (HF), TRL, axolotl, LLaMA-Factory, unsloth (2-5x speedup), litGPT. Decision: full FT (best quality, +memory), DoRA (quality + memory), QLoRA (low GPU, custom domain), LoRA (production + speed). Memory: 65B QLoRA 48GB, 7B QLoRA 16GB. Hoy: QLoRA + DoRA + unsloth es SOTA production.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/06-instruction-tuning-sft, 10/08-dpo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LoRA init y forward.
- Implementar LoRA merge.
- Comparar LoRA vs QLoRA vs DoRA params.
- Diagnosticar frameworks y decision criteria.

## Constrúyelo

```python
def lora_init(in_dim, out_dim, rank=4, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((in_dim, rank))
    B = np.zeros((rank, out_dim))
    return A, B
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: fine-tuning-lora
fase: 11
leccion: 08
---

1. LoRA W + alpha/r * A @ B.
2. QLoRA 4-bit + LoRA.
3. DoRA, rsLoRA, LongLoRA.
4. PEFT, TRL, unsloth.
5. QLoRA default.
```

## Ejercicios

1. **LoRA**: implementar y entrenar
   LoRA en custom data.
2. **QLoRA**: fine-tune 7B
   con QLoRA en RTX 4090.
3. **Desafio**: DoRA + DPO
   combo.

## Lecturas recomendadas

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)
- "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models" (Chen et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Fine Tuning LoRA]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).