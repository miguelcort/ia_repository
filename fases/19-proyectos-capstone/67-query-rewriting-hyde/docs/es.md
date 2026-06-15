# 67 — Query rewriting: HyDE

> Query rewriting: reformular query para mejor retrieval. HyDE (Hypothetical Document Embeddings): LLM genera respuesta hipotética, embed, retrieve. Multi-query: LLM genera variantes, retrieve todas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/65, 19/66
**Tiempo estimado:** ~20 minutos

## Objetivos

- HyDE.
- Multi-query.
- Step-back prompting.
- RAG-Fusion.

## Constrúyelo

```python
def hyde_retrieval(query, llm, retriever, k=10):
    """HyDE: hypothetical document, then retrieve."""
    # 1. Generate hypothetical answer
    hyp = llm(f"Answer concisely: {query}")
    # 2. Embed hypothetical answer
    # 3. Retrieve with this embedding
    return retriever.query(hyp, top_k=k)


def multi_query(query, llm, retriever, n=4):
    """Generate variants, retrieve all, RRF."""
    variants = llm(f"Generate {n} alternative phrasings: {query}")
    results = [retriever.query(v) for v in variants]
    return rrf_fuse(results)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-query-rewrite
fase: 19
leccion: 67
---

1. HyDE.
2. Multi-query.
3. Step-back.
4. RAG-Fusion.
```

## Ejercicios

1. **HyDE**: 100 queries.
2. **Multi-query**: 4
   variants.
3. **Desafío**: +10%
   recall.

## Lecturas recomendadas

- "HyDE" (Gao 2022)
- "RAG-Fusion" (2024)
- "Step-back" (Zheng 2023)



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
> "[67-query-rewriting-hyde]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
