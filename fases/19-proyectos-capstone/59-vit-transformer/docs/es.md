# 59 — ViT (Vision Transformer)

> ViT (Dosovitskiy 2020): transformer puro para visión. Patches como tokens, positional embeddings, encoder. Pre-training: supervised (ImageNet) o self-supervised (DINO, MAE, BEiT).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/58
**Tiempo estimado:** ~25 minutos

## Objetivos

- ViT architecture.
- Pre-training (DINO, MAE).
- Fine-tuning.
- Compare con CNN.

## Constrúyelo

```python
import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                d=768, n_heads=12, n_layers=12, n_classes=1000):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch_size, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vit
fase: 19
leccion: 59
---

1. ViT.
2. Pre-training.
3. Fine-tune.
4. DINO / MAE.
```

## Ejercicios

1. **ViT-tiny**: 6 layers.
2. **Pre-train** DINO.
3. **Desafío**: ViT-L
   en ImageNet.

## Lecturas recomendadas

- "ViT" (Dosovitskiy 2020)
- "DINO" (Caron 2021)
- "MAE" (He 2021)
- "BEiT" (Bao 2021)



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
> "[59-vit-transformer]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
