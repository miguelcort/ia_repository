# Modelos de embedding a profundidad

> Vectores de frases para semantic search, RAG, clustering. Contrastive learning (InfoNCE) + hard negative mining. SOTA 2024: bge-large, e5-large, gte-large, OpenAI text-embedding-3. Frameworks: sentence-transformers, MTEB benchmark.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 21-inferencia-de-texto-nli
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar contrastive loss (InfoNCE).
- Implementar mean/CLS pooling.
- Implementar hard negative mining.
- Diagnosticar sentence embedding SOTA.

## Constrúyelo

```python
def contrastive_loss(z1, z2, t=0.07):
    z = np.vstack([z1, z2])
    sim = z @ z.T / t
    exp = np.exp(sim - sim.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    loss = 0
    for i in range(len(z1)):
        loss -= np.log(probs[i, i + len(z1)] + 1e-10)
    return loss / (2 * len(z1))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-embeddings
fase: 05
leccion: 22
---

1. Open SOTA: bge-large, e5-large, gte-large.
2. Multilingual: bge-m3, mE5-large.
3. Closed: OpenAI text-embedding-3, Cohere v3.
4. Mobile: all-MiniLM-L6-v2.
5. Custom: fine-tune con 10-100K pares.
6. MTEB eval, FAISS para 1M+ docs.
```

## Ejercicios

1. **MultipleNegativesRankingLoss**: implementar la loss de
   sentence-transformers.
2. **Hard negative mining**: implementar mining con
   BM25 + dense bi-encoder.
3. **Desafio**: fine-tunear bge-small-es sobre un dataset
   custom (e.g. preguntas y respuestas internas),
   mejorar MTEB retrieval en 5+ puntos.

## Lecturas recomendaciones

- "SBERT" (Reimers & Gurevych, 2019)
- "BGE" (BAAI, 2023): <https://huggingface.co/BAAI/bge-large-en-v1.5>
- sentence-transformers: <https://sbert.net/>
- MTEB: <https://huggingface.co/spaces/mteb/leaderboard>

---

> 📚 **Adaptación al español** de la lección "[Embedding Models Deep Dive]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).