# 51 — Literature retrieval

> Literature retrieval: búsqueda en arXiv, Semantic Scholar, OpenAlex, PubMed. Pipeline: query expansion, multi-source search, re-ranking, dedup. Citations, abstracts, full-text.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/50
**Tiempo estimado:** ~25 minutos

## Objetivos

- Multi-source search.
- Re-ranking.
- Citation graph.
- Full-text access.

## Constrúyelo

```python
import arxiv
from semanticscholar import SemanticScholar


def multi_source_search(query, n=20):
    """Search arXiv + Semantic Scholar."""
    arxiv_results = arxiv.Search(query, max_results=n)
    s2 = SemanticScholar()
    s2_results = s2.search_paper(query, limit=n)
    return {
        "arxiv": [r.title for r in arxiv_results.results()],
        "semantic_scholar": [p.title for p in s2_results],
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
name: prompt-lit-search
fase: 19
leccion: 51
---

1. Multi-source (arXiv, S2, OpenAlex).
2. Query expansion.
3. Re-ranking.
4. Citation graph.
5. Dedup.
```

## Ejercicios

1. **Search**: 10 topics.
2. **Re-rank**: cross-
   encoder.
3. **Desafío**: full
   corpus de 1K papers.

## Lecturas recomendadas

- "arXiv API" (2024)
- "Semantic Scholar" (Allen AI)
- "OpenAlex" (2022)



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
> "[51-literature-retrieval]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
