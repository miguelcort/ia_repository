# RAG (Retrieval-Augmented Generation)

> RAG (Lewis 2020, Meta): retrieve docs, augment prompt, generate. Pipeline: document loading (PDF, HTML) → chunking (256-1024 tokens, 10-20% overlap) → embedding (BGE-M3, OpenAI text-embedding-3, E5) → vector store (Pinecone, Weaviate, Qdrant, Milvus, Chroma, pgvector) → retrieval (top-k) → re-ranking (bge-reranker-v2, Jina, Cohere, +5-15% recall) → LLM generation. Variantes: Naive, Advanced (re-ranking, query rewriting, hybrid), Modular, GraphRAG (Microsoft 2024, graph-based, communities), Self-RAG (Asai 2023, UW, LLM self-critique), CRAG (Yan 2024, corrective), Agentic RAG (multi-step, tools). Frameworks: LlamaIndex, LangChain, Haystack, DSPy, mem0. Eval: RAGAS (faithfulness, answer relevance, context relevance, context recall), TruLens, DeepEval, LangSmith. Hoy: modular + agentic RAG es SOTA.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/04-embeddings
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar vector store.
- Implementar retrieve top-k.
- Implementar build RAG prompt.
- Diagnosticar pipelines y eval.

## Constrúyelo

```python
def retrieve(query_emb, store, top_k=3):
    sims = [(doc, cosine_sim(query_emb, emb)) for doc, emb in store.items()]
    sims.sort(key=lambda x: -x[1])
    return [doc for doc, _ in sims[:top_k]]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: rag
fase: 11
leccion: 06
---

1. Index docs, retrieve, augment.
2. Vector store, re-ranker.
3. Modular, GraphRAG, Self-RAG.
4. LlamaIndex, LangChain.
5. RAGAS eval.
```

## Ejercicios

1. **RAG**: implementar pipeline
   completo.
2. **GraphRAG**: graph-based
   community detection.
3. **Desafio**: agentic RAG
   con multi-hop.

## Lecturas recomendadas

- "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al., 2023)
- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023)
- "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft, 2024)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[RAG]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).