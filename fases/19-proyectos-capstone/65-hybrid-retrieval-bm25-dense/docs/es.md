# 65 — Hybrid retrieval: BM25 + dense

> Hybrid retrieval: combina BM25 (sparse, lexical) + dense (semantic embeddings). Score = α·BM25 + β·dense. RRF (Reciprocal Rank Fusion) o convex combination. Mejora recall sobre dense-only.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/64
**Tiempo estimado:** ~25 minutos

## Objetivos

- BM25 index.
- Dense embeddings.
- Score fusion.
- Eval retrieval quality.

## Constrúyelo

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


class HybridRetriever:
    def __init__(self, docs, alpha=0.5):
        self.bm25 = BM25Okapi([d.split() for d in docs])
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeds = self.model.encode(docs)
        self.docs = docs
        self.alpha = alpha

    def query(self, q, top_k=10):
        # BM25
        bm25_scores = self.bm25.get_scores(q.split())
        # Dense
        q_embed = self.model.encode([q])
        dense_scores = (self.doc_embeds @ q_embed.T).flatten()
        # Normalize
        bm25_scores = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min() + 1e-8)
        dense_scores = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-8)
        # Hybrid
        scores = self.alpha * bm25_scores + (1 - self.alpha
                                            ) * dense_scores
        idx = np.argsort(-scores)[:top_k]
        return [(self.docs[i], scores[i]) for i in idx]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-hybrid
fase: 19
leccion: 65
---

1. BM25 + dense.
2. RRF / score fusion.
3. Eval MRR, nDCG.
```

## Ejercicios

1. **Hybrid**: 10K docs.
2. **RRF**: rank fusion.
3. **Desafío**: +15%
   recall sobre dense.

## Lecturas recomendadas

- "BM25" (Robertson 2009)
- "RRF" (Cormack 2009)
- "Hybrid Search" (2023)

---

> 📚 **Adaptación al español** de la lección
> "[65-hybrid-retrieval-bm25-dense]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
