# 54 — Paper writer

> Paper writer: LLM genera draft de paper académico. Sections: abstract, intro, related, method, results, discussion. Citations, figures, tables. Frameworks: PaperDigest, ResearchAgent, scite.ai.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/50-53
**Tiempo estimado:** ~25 minutos

## Objetivos

- Section generation.
- Citation insertion.
- Figure/table generation.
- LaTeX output.

## Constrúyelo

```python
def write_paper_sections(results, related_work, llm):
    """Generate paper sections."""
    abstract = llm(f"Write abstract for: {results}")
    intro = llm(f"Write intro citing: {related_work}")
    method = llm(f"Write methods for: {results['method']}")
    results_text = llm(f"Write results: {results['data']}")
    discussion = llm(f"Write discussion: {results['conclusions']}")
    return {"abstract": abstract, "intro": intro,
            "method": method, "results": results_text,
            "discussion": discussion}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-paper-writer
fase: 19
leccion: 54
---

1. Section generation.
2. Citations.
3. Figures / tables.
4. LaTeX.
```

## Ejercicios

1. **Generate**: 1 paper
   draft.
2. **Insert citations**:
   30 references.
3. **Desafío**: compile
   LaTeX.

## Lecturas recomendadas

- "AI Co-scientist" (Google 2024)
- "ResearchAgent" (2024)
- "scite.ai" (2024)



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
> "[54-paper-writer]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
