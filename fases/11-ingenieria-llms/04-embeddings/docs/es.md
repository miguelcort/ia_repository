# Embeddings

> Embeddings: vectores densos R^d que representan texto. Word (Word2Vec, GloVe), Sentence (SBERT, E5, BGE), Document (long-context, late-interaction ColBERT). Cosine sim: dot / (|a| · |b|). MTEB benchmark (Hugging Face, 100+ tasks, leaderboard). SOTA 2024-25: OpenAI text-embedding-3-large (3072 dim, MTEB 64), BGE-M3 (multilingual, 8K ctx), E5-Mistral-7B-instruct, GTE-Qwen2-7B-instruct, Voyage-3, Cohere embed-v3. ColBERTv2 (Khattab 2020, Stanford): late-interaction per-token, +5-10% vs bi-encoder, PLAID index fast retrieval. ColPali (2024): vision RAG. Hybrid: dense + sparse (BM25) + cross-encoder rerank.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/06-rag (próximo)
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar cosine similarity.
- Implementar L2 normalize.
- Implementar top-k similar.
- Diagnosticar modelos SOTA.

## Constrúyelo

```python
def cosine_similarity(a, b):
    a_n = normalize(a)
    b_n = normalize(b)
    return float(np.dot(a_n, b_n))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: embeddings
fase: 11
leccion: 04
---

1. Word, sentence, document.
2. Cosine sim.
3. MTEB benchmark.
4. BGE-M3, OpenAI, ColBERTv2.
5. Hybrid + rerank.
```

## Ejercicios

1. **Embeddings**: comparar
   SBERT vs OpenAI en MTEB.
2. **ColBERTv2**: implementar
   ColBERT simple.
3. **Desafio**: hybrid retrieval
   dense + BM25 + rerank.

## Lecturas recomendadas

- "Sentence-BERT: Sentence Embeddings using Siamese BERT" (Reimers & Gurevych, 2019)
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (Khattab & Zaharia, 2020)
- "MTEB: Massive Text Embedding Benchmark" (Muennighoff et al., 2022)
- "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings" (Chen et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Embeddings]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).