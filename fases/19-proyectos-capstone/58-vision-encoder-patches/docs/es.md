# 58 — Vision encoder: patches

> Vision encoder (ViT, Dosovitskiy 2020): split image en patches (16x16), linear projection, positional embeddings, transformer encoder. Standard para vision-language models (CLIP, LLaVA, Qwen-VL).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 12 (multimodal)
**Tiempo estimado:** ~25 minutos

## Objetivos

- Patch embedding.
- Positional embedding.
- ViT architecture.
- Compare con CNN.

## Constrúyelo

```python
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, img_size=224, patch_size=16,
                d_model=768):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size,
                             stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_patches, d_model))

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)  # (B, d, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d)
        return x + self.pos
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vit-patches
fase: 19
leccion: 58
---

1. Patch extraction.
2. Linear projection.
3. Positional embedding.
4. ViT.
```

## Ejercicios

1. **Patch**: 16x16.
2. **ViT-tiny**: 6 layers.
3. **Desafío**: ViT-L
   con 300M params.

## Lecturas recomendadas

- "ViT" (Dosovitskiy 2020)
- "DeiT" (Touvron 2021)
- "Swin" (Liu 2021)



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
> "[58-vision-encoder-patches]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
