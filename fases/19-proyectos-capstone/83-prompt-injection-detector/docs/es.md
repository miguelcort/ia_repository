# 83 — Prompt injection detector

> Prompt injection detector: classifier que distingue instruction vs data. Train sobre data con etiquetas injection/no-injection. F1 ~95% en test set. DeBERTa, GPT-4 judge, regex rules.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 18/15, 19/82
**Tiempo estimado:** ~20 minutos

## Objetivos

- Train classifier.
- Regex + ML hybrid.
- Threshold tuning.
- Eval sobre BIPIA.

## Constrúyelo

```python
from transformers import pipeline


def injection_detector(text, classifier):
    """Detect prompt injection."""
    result = classifier(text)
    return {"is_injection": result[0]["label"] == "INJECTION",
            "score": result[0]["score"]}


def regex_injection_check(text):
    """Simple regex-based check."""
    patterns = [r"ignore previous",
               r"system override",
               r"new instructions",
               r"forget everything"]
    return any(__import__("re").search(p, text.lower())
              for p in patterns)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-injection-detector
fase: 19
leccion: 83
---

1. Classifier.
2. Regex + ML.
3. Threshold.
4. Eval BIPIA.
```

## Ejercicios

1. **Train**: 1K
   injection examples.
2. **Eval**: BIPIA
   dataset.
3. **Desafío**: F1 >
   0.95.

## Lecturas recomendadas

- "BIPIA" (Yi 2023)
- "PromptInjection"
  (2024)
- "DeBERTa-v3" (He 2021)



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
> "[83-prompt-injection-detector]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
