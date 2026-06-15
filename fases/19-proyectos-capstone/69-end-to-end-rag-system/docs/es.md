# 69 — End-to-end RAG system

> End-to-end RAG: pipeline completo, AST chunking + hybrid retrieval + re-rank + citations + LLM synthesis + monitoring. Production-grade. Multitenant, low latency, observability.

**Tipo:** Capstone
**Lenguajes:** Python (backend), TypeScript (UI)
**Prerrequisitos:** Fase 19/64-68, Fase 17
**Tiempo estimado:** 30 horas

## Objetivos

- Full RAG pipeline.
- Production observability.
- Multi-tenant.
- Eval + monitoring.

## Constrúyelo

```python
class ProductionRAG:
    def __init__(self, llm, embedder, reranker, retriever):
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.retriever = retriever

    def query(self, q, top_k=10, return_citations=True):
        # 1. HyDE
        hyp = self.llm.hyde(q)
        # 2. Hybrid retrieval
        candidates = self.retriever.query(hyp, top_k=50)
        # 3. Re-rank
        ranked = self.reranker.rerank(q, candidates, top_k=5)
        # 4. LLM synthesis
        answer = self.llm.synthesize(q, ranked)
        return {
            "answer": answer,
            "citations": [r["id"] for r in ranked]
                if return_citations else None,
        }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-e2e-rag
fase: 19
leccion: 69
---

1. HyDE + hybrid.
2. Re-rank.
3. Synthesis + citations.
4. Observability.
5. Multi-tenant.
```

## Ejercicios

1. **Pipeline**: 10K docs.
2. **Eval**: RAGAS suite.
3. **Desafío**: production
   deploy.

## Lecturas recomendadas

- "RAGAS" (Es 2023)
- "Production RAG"
  (LangChain 2024)
- "Haystack" (deepset 2023)



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
> "[69-end-to-end-rag-system]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
