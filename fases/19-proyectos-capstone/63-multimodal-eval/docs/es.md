# 63 — Multimodal eval

> Multimodal eval: VQA, image captioning, multimodal reasoning, OCR, chart understanding. Benchmarks: MMBench, MMMU, MathVista, ChartQA, DocVQA. Frameworks: lm-eval-harness (multi-modal), VLMEvalKit.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/62
**Tiempo estimado:** ~25 minutos

## Objetivos

- MMBench / MMMU.
- MathVista / ChartQA.
- VLMEvalKit.
- Custom multimodal eval.

## Constrúyelo

```python
def vqa_eval(model, dataset):
    """VQA accuracy."""
    correct = 0
    for item in dataset:
        pred = model(image=item["image"],
                    question=item["question"])
        if pred == item["answer"]:
            correct += 1
    return correct / len(dataset)


def mm_bench_eval(model):
    """MMBench: multiple choice multimodal."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MMBench")
    return vqa_eval(model, ds["test"])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mm-eval
fase: 19
leccion: 63
---

1. MMBench / MMMU.
2. MathVista.
3. ChartQA.
4. VLMEvalKit.
```

## Ejercicios

1. **MMBench**: GPT-4V
   vs Qwen-VL.
2. **MathVista**: math
   reasoning.
3. **Desafío**: custom
   domain eval.

## Lecturas recomendadas

- "MMBench" (Liu 2024)
- "MMMU" (Yue 2024)
- "VLMEvalKit" (OpenCompass)
- "MathVista" (Lu 2023)



## Detalles avanzados

Esta lección cubre los trade-offs críticos de
producción. Considera scaling: en pre-training el
factor dominante es cómputo disponible; en inference
es latencia y costo. Frameworks standard: PyTorch
(HF Transformers, TRL, vLLM), JAX (Flax, Optax).
Optimizaciones: FlashAttention-2, paged attention,
KV cache compression, speculative decoding, MoE.

Eval riguroso: statistical significance testing
sobre múltiples seeds, held-out test sets sin
contamination, y edge cases del domain. Métricas:
BLEU/ROUGE para text generation, exact match/F1
para QA, pass@k para code, human preference para
chat.

Trampas comunes: data leakage entre train/test,
overfitting al validation set, eval con prompts
fuera de distribución, ignore de tail latency en
serving, cost runaway en production.

Tools clave: Weights & Biases o MLflow para
tracking, Langfuse para LLM observability, Hydra
para config, Ray para distributed execution, vLLM
para serving LLM. Conoce al menos uno a fondo antes
de producción.

---

> 📚 **Adaptación al español** de la lección
> "[63-multimodal-eval]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
