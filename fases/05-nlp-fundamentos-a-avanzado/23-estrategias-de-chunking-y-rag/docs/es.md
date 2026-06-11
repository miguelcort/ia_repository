# Estrategias de chunking y RAG

> Dividir docs en chunks para embedding y retrieval. Fixed (256-512 tokens, overlap 10-20%), sentence, semantic, hierarchical. Embeddings SOTA (BGE, OpenAI), ANN (FAISS), reranker, LLM con contexto. Frameworks: Haystack, LlamaIndex, LangChain, RAGFlow.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 22-modelos-de-embedding-a-profundidad
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar chunking fijo, sentence, semantic.
- Medir overlap entre chunks.
- Diagnosticar estrategias segun tipo de doc.
- Diagnosticar hierarchical chunking.

## Constrúyelo

```python
def chunking_fijo(texto, chunk_size=200, overlap=50):
    palabras = texto.split()
    chunks = []
    i = 0
    while i < len(palabras):
        chunks.append(" ".join(palabras[i:i + chunk_size]))
        if i + chunk_size >= len(palabras): break
        i += chunk_size - overlap
    return chunks
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rag
fase: 05
leccion: 23
---

1. Default: chunking 512, BGE/OpenAI, FAISS, LLM.
2. Custom: parser + chunking document-aware.
3. Hybrid: BM25 + dense + RRF.
4. Long context: hierarchical, parent-child.
5. Eval: RAGAS, TruLens.
6. Frameworks: Haystack, LlamaIndex, LangChain.
```

## Ejercicios

1. **Semantic chunker**: implementar Greg Kamradt's
   semantic chunker.
2. **ParentDocumentRetriever**: implementar jerarquico
   con parent + child.
3. **Desafio**: pipeline RAG end-to-end con LlamaParse +
   BGE + FAISS + Claude + citations, evaluar con RAGAS
   en un dataset custom.

## Lecturas recomendadas

- "Semantic Chunking" (Greg Kamradt, 2024)
- "Parent Document Retriever" (LangChain)
- RAGAS: <https://docs.ragas.io/>
- RAGFlow: <https://github.com/infiniflow/ragflow>

---

> 📚 **Adaptación al español** de la lección "[Chunking Strategies & RAG]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).