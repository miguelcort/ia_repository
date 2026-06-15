# 64 — Chunking strategies advanced

> Chunking strategies: fixed-size, sentence, semantic, AST-aware, hierarchical. Trade-offs: granularity, context preservation, embedding quality. Frameworks: LangChain, LlamaIndex, semantic-chunker.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/08
**Tiempo estimado:** ~25 minutos

## Objetivos

- Semantic chunking.
- AST-aware (code).
- Hierarchical.
- Eval retrieval quality.

## Constrúyelo

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def semantic_chunk(text, model, threshold=0.7):
    """Split text en chunks con similaridad > threshold."""
    sentences = text.split(". ")
    model = SentenceTransformer(model)
    embeddings = model.encode(sentences)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = embeddings[i] @ embeddings[i - 1] / (
            np.linalg.norm(embeddings[i])
            * np.linalg.norm(embeddings[i - 1]))
        if sim > threshold:
            chunks[-1].append(sentences[i])
        else:
            chunks.append([sentences[i]])
    return [". ".join(c) for c in chunks]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-chunking
fase: 19
leccion: 64
---

1. Fixed vs semantic.
2. AST-aware (code).
3. Hierarchical.
4. Eval retrieval.
```

## Ejercicios

1. **Semantic**: 100 docs.
2. **AST**: Python code.
3. **Desafío**: hybrid
   chunker.

## Lecturas recomendadas

- "Chunking Strategies"
  (Greg 2024)
- "LangChain" (Chase 2023)
- "LlamaIndex" (Liu 2022)



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
> "[64-chunking-strategies-advanced]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
