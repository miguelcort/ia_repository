# RAG avanzado

> Advanced RAG: HyDE (Hypothetical Document Embeddings, +5-10% recall), multi-query (reformulación), step-back, query decomposition, hybrid (BM25 + dense), RRF (Reciprocal Rank Fusion, k=60), cross-encoder re-ranking (bge-reranker-v2, Jina, Cohere, +5-15%), Self-RAG (Asai 2023, UW, LLM self-critique con tokens `<retrieve>`/`<relevant>`), CRAG (Yan 2024, confidence-based corrective), GraphRAG (Microsoft 2024, graph + communities + retrieval). Frameworks: LlamaIndex query engines, LangChain re-rankers, RAGatouille, Haystack, DSPy, GraphRAG. +10-30% quality. SOTA 2024-25: modular + agentic + hybrid + re-rank + ColPali (vision RAG, Faysse 2024) + reasoning (o1-style search + RAG).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/06-rag
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar HyDE hypothetical.
- Implementar multi-query.
- Implementar RRF.
- Diagnosticar Self-RAG y CRAG.

## Constrúyelo

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: advanced-rag
fase: 11
leccion: 07
---

1. HyDE, multi-query, RRF.
2. Hybrid + re-rank.
3. Self-RAG, CRAG.
4. GraphRAG.
5. +10-30% quality.
```

## Ejercicios

1. **HyDE**: implementar HyDE
   en LlamaIndex.
2. **CRAG**: implementar
   confidence-based corrective.
3. **Desafio**: GraphRAG
   en custom corpus.

## Lecturas recomendadas

- "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, Gao et al., 2022)
- "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
- "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
- "From Local to Global: GraphRAG" (Microsoft, 2024)
- "Reciprocal Rank Fusion" (Cormack et al., 2009)

---

> 📚 **Adaptación al español** de la lección "[Advanced RAG]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).