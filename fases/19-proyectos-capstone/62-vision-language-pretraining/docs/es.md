# 62 — Vision-language pretraining

> Vision-language pretraining: contrastive (CLIP, SigLIP, SigLIP 2) o generative (LLaVA, Qwen-VL). CLIP: contrastive entre image-text pairs, 400M pairs. SigLIP: sigmoid loss, mejor scaling.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/60, 19/61
**Tiempo estimado:** ~30 minutos

## Objetivos

- CLIP contrastive loss.
- SigLIP sigmoid loss.
- Image-text dataset.
- Zero-shot transfer.

## Constrúyelo

```python
import torch
import torch.nn.functional as F


def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """CLIP contrastive loss."""
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(logits.size(0))
    return (F.cross_entropy(logits, labels)
           + F.cross_entropy(logits.T, labels)) / 2


def siglip_loss(image_embeds, text_embeds, temperature=10.0):
    """SigLIP sigmoid loss."""
    logits = image_embeds @ text_embeds.T * temperature
    labels = torch.eye(logits.size(0)) * 2 - 1
    return -F.logsigmoid(labels * logits).sum() / logits.size(0)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vl-pretrain
fase: 19
leccion: 62
---

1. Image-text pairs.
2. Contrastive (CLIP).
3. Sigmoid (SigLIP).
4. Zero-shot eval.
```

## Ejercicios

1. **CLIP**: 10K pairs.
2. **SigLIP**: 100K.
3. **Desafío**: zero-
   shot ImageNet 1K.

## Lecturas recomendadas

- "CLIP" (Radford 2021)
- "SigLIP" (Zhai 2023)
- "SigLIP 2" (Google 2025)



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
> "[62-vision-language-pretraining]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
