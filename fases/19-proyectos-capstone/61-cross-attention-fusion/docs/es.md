# 61 — Cross-attention fusion

> Cross-attention fusion (Flamingo, BLIP-2): LLM attends to vision features via cross-attention. Más expresivo que linear projection. Q-Former (BLIP-2): small transformer que extrae queries.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/60
**Tiempo estimado:** ~25 minutos

## Objetivos

- Q-Former architecture.
- Cross-attention layers.
- Vision-text alignment.
- Compare con linear.

## Constrúyelo

```python
import torch
import torch.nn as nn


class QFormer(nn.Module):
    """Q-Former: queries attend to vision features."""
    def __init__(self, n_queries=32, vision_dim=1024,
                d=768, n_layers=6, n_heads=12):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(1, n_queries, d))
        self.vision_proj = nn.Linear(vision_dim, d)
        self.layers = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])

    def forward(self, vision_features):
        # vision_features: (B, N_v, vision_dim)
        v = self.vision_proj(vision_features)
        q = self.queries.expand(v.size(0), -1, -1)
        for layer in self.layers:
            q = layer.cross_attn(q, v, v)
        return q
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-cross-attn
fase: 19
leccion: 61
---

1. Q-Former.
2. Cross-attention.
3. Vision-text.
4. Compare con linear.
```

## Ejercicios

1. **Q-Former**: 32
   queries.
2. **Train**: image-text.
3. **Desafío**: BLIP-2
   style.

## Lecturas recomendadas

- "BLIP-2" (Li 2023)
- "Flamingo" (DeepMind 2022)
- "Q-Former" (Li 2023)



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
> "[61-cross-attention-fusion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
