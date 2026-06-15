# 08 — Production RAG chatbot

> Production RAG: chunking (semantic, AST), hybrid retrieval (BM25 + dense), re-ranking (cross-encoder), query rewriting (HyDE), evaluation (RAGAS, TruLens). Monitoreo, caching, cost control. Cita verificables.

**Tipo:** Capstone
**Lenguajes:** Python (backend), TypeScript (UI)
**Prerrequisitos:** Fase 11 (LLM), Fase 13 (tools), Fase 17
**Tiempo estimado:** 25 horas

## Objetivos

- Pipeline RAG production-grade.
- Hybrid retrieval + re-ranking.
- Citation faithfulness.
- Eval (RAGAS) + monitoring.

## El problema

RAG en producción (Sourcegraph Cody, Notion AI,
Slack AI): (1) Ingestión AST-aware (tree-sitter).
(2) Chunking semántico (semantic chunker, 256-512
tokens). (3) Hybrid retrieval (BM25 + dense
embeddings). (4) Re-ranking (cross-encoder).
(5) Query rewriting (HyDE, multi-query). (6) LLM
synthesis con citations. (7) Eval: RAGAS
(context_precision, faithfulness, answer_relevancy).
(8) Monitoring: token cost, latency, retrieval
quality.

## Constrúyelo

```python
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder


class ProductionRAG:
    def __init__(self, llm, embedder, reranker):
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker

    def query(self, question, top_k=10):
        # Hybrid retrieval
        bm25 = self.bm25_retrieve(question, k=top_k)
        dense = self.dense_retrieve(question, k=top_k)
        candidates = merge_unique(bm25, dense)
        # Re-rank
        scores = self.reranker.predict(
            [(question, c.text) for c in candidates])
        ranked = sorted(zip(candidates, scores),
                       key=lambda x: -x[1])[:5]
        # Synthesize with citations
        return self.llm.synthesize(question, ranked)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-production-rag
fase: 19
leccion: 08
---

1. AST chunking.
2. Hybrid retrieval.
3. Cross-encoder re-rank.
4. Citations.
5. RAGAS eval.
```

## Ejercicios

1. **RAG**: ingestar 10K docs,
   benchmark retrieval.
2. **Eval**: correr RAGAS suite.
3. **Desafío**: multi-tenant
   production deploy.

## Lecturas recomendadas

- "RAGAS" (Es 2023)
- "TruLens" (2024)
- "LangChain" (Chase 2023)
- "ColBERT" (Khattab 2020)

---

> 📚 **Adaptación al español** de la lección
> "[08-production-rag-chatbot]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
