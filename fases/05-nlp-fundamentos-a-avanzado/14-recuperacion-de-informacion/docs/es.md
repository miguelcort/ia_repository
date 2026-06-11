# Recuperación de información

> Dada una query, encontrar documentos relevantes. Lexical (BM25, TF-IDF) y dense (DPR, ColBERT) son las dos familias. Hybrid + reranker es el estado del arte en produccion.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13-preguntas-y-respuestas
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Construir indice invertido.
- Implementar BM25 con IDF y normalizacion por longitud.
- Evaluar con Precision@K.
- Diagnosticar lexical vs dense vs hybrid.

## Constrúyelo

```python
def bm25(query, docs, indice, k1=1.5, b=0.75):
    N = len(docs)
    doc_lens = [len(tokenizar(d)) for d in docs]
    avgdl = sum(doc_lens) / N
    scores = np.zeros(N)
    for t in tokenizar(query):
        if t not in indice: continue
        df = len(indice[t])
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for doc_id, posiciones in indice[t].items():
            f = len(posiciones)
            denom = f + k1 * (1 - b + b * doc_lens[doc_id] / avgdl)
            scores[doc_id] += idf * f * (k1 + 1) / denom
    return scores
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ir
fase: 05
leccion: 14
---

1. Search: Elasticsearch + vector plugin, Vespa.
2. RAG: BM25 + ColBERT + reranker + LLM.
3. Custom: dense bi-encoder fine-tune.
4. Multilingual: ColBERT-X o dense multilingual.
5. Zero-shot: BGE, E5, GTE.
6. nDCG@10 + Recall@100.
```

## Ejercicios

1. **BM25 parametros**: explorar k1, b y ver impacto en
   nDCG.
2. **Hybrid**: implementar Reciprocal Rank Fusion (RRF)
   entre BM25 y dense.
3. **Desafio**: indexar 100K documentos con BM25 + dense,
   comparar Recall@10 y latencia.

## Lecturas recomendadas

- "BM25" (Robertson et al., 1994)
- "Dense Passage Retrieval for Open-Domain QA" (Karpukhin
  et al., 2020)
- "ColBERTv2" (Santhanam et al., 2022)
- Elasticsearch: <https://www.elastic.co/>

---

> 📚 **Adaptación al español** de la lección "[Information Retrieval]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).