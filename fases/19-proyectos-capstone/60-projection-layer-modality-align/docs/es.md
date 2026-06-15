# 60 — Projection layer: modality align

> Projection layer (LLaVA, Qwen-VL): vision features (ViT) → linear projection → LLM embedding space. Critical para vision-language models. Entrenado sobre image-text pairs.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/58, 19/59
**Tiempo estimado:** ~20 minutos

## Objetivos

- Linear projection.
- MLP projector.
- Image-text alignment.
- Frozen ViT + trainable.

## Constrúyelo

```python
import torch.nn as nn


class VisionProjector(nn.Module):
    """Project vision features to LLM embedding space."""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features):
        return self.proj(vision_features)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vision-proj
fase: 19
leccion: 60
---

1. Frozen ViT.
2. Trainable projector.
3. LLM input: tokens +
   image embeddings.
4. Image-text alignment.
```

## Ejercicios

1. **Projector**: ViT-L →
   Llama 7B dim.
2. **Train**: 100K image-
   text pairs.
3. **Desafío**: full VLM
   training.

## Lecturas recomendadas

- "LLaVA" (Liu 2023)
- "Qwen-VL" (Bai 2023)
- "InstructBLIP" (Dai 2023)



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
> "[60-projection-layer-modality-align]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
