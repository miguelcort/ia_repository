# 66 — Reranker cross-encoder

> Cross-encoder re-ranker (Cohere Rerank, monoT5, BGE reranker, Jina): dado (query, doc), score con cross-attention. Más preciso que bi-encoder, más caro. Top-K re-rank.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/65
**Tiempo estimado:** ~25 minutos

## Objetivos

- Cross-encoder architecture.
- Top-K re-ranking.
- Cohere Rerank API.
- BGE / Jina reranker.

## Constrúyelo

```python
from sentence_transformers import CrossEncoder


def cross_encoder_rerank(query, docs, top_k=5,
                       model="BAAI/bge-reranker-v2-m3"):
    reranker = CrossEncoder(model)
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked[:top_k]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rerank
fase: 19
leccion: 66
---

1. Cross-encoder.
2. Top-K re-rank.
3. Cohere Rerank API.
4. Eval nDCG.
```

## Ejercicios

1. **Re-rank**: 100
   candidates.
2. **Compare**: BGE,
   Jina, Cohere.
3. **Desafío**: +20%
   nDCG@10.

## Lecturas recomendadas

- "monoT5" (Nogueira 2020)
- "BGE Reranker" (BAAI 2023)
- "Cohere Rerank" (2024)



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
> "[66-reranker-cross-encoder]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
