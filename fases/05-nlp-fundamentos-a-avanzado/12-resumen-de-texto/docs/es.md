# 12 — Resumen de texto

> Resumir un documento en uno o dos párrafos. Abstractive summarization (generar texto nuevo) es más desafiante que extractive (seleccionar oraciones).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-traduccion-automatica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar extractive summarization con TextRank.
- Aplicar modelos abstractive (BART, T5, PEGASUS, GPT-4).
- Diagnosticar métricas: ROUGE, BERTScore, BARTScore.
- Conocer las técnicas modernas: long-context,
  hierarchical, controlled summarization.

## El problema

Resumir un documento preservando la información esencial
es una tarea central de NLP. Extractive summarization
selecciona las oraciones más importantes del original.
Abstractive summarization genera texto nuevo, más natural
pero más difícil. Los LLMs modernos hacen abstractive de
calidad con prompting adecuado.

## El concepto

**Extractive vs Abstractive.**

- **Extractive:** selecciona oraciones o frases del
  documento. Más fácil, más factual, menos natural.
  TextRank, LexRank, BERTSum-extractive.
- **Abstractive:** genera texto nuevo. Más natural,
  puede reformular, pero puede "alucinar" contenido
  que no está en el original. BART, T5, PEGASUS,
  GPT-4.

**TextRank (Mihalcea & Tarau, 2004).** Algoritmo de
graph-based ranking para extractive summarization:

1. Construye un grafo donde cada oración es un nodo.
2. Añade aristas entre oraciones según su similitud
   (cosine de TF-IDF o embeddings).
3. Aplica PageRank para puntuar cada oración.
4. Selecciona las top-K oraciones en el orden original.

**Modelos abstractive.**

- **BART (Lewis et al., 2020):** encoder-decoder
  transformer denoising autoencoder. Pre-entrenado
  corrompiendo texto y aprendiendo a reconstruir.
- **T5 (Raffel et al., 2020):** transformer
  encoder-decoder "text-to-text". Toda tarea es
  reformulada como text-to-text.
- **PEGASUS (Zhang et al., 2020):** pre-entrenado con
  Gap Sentence Generation (esconder frases y predecirlas).
  SOTA en resumen.
- **FLAN-T5, FLAN-PaLM:** instruction-tuned, siguen
  siendo fuertes.

**Métricas de evaluación.**

- **ROUGE (Recall-Oriented Understudy for Gisting
  Evaluation):** mide n-gramas en común. ROUGE-1,
  ROUGE-2, ROUGE-L.
- **BERTScore:** similitud contextual entre
  embeddings de candidato y referencia.
- **BARTScore:** usa un BART fine-tuneado para
  predecir la "calidad" del resumen.
- **LLM-as-judge:** GPT-4 como juez (con caveats).

**Pipeline típico en producción.**

1. Pre-procesar: dividir documento largo en chunks
   (< 8k tokens para GPT-3.5, 128k para GPT-4).
2. Resumir cada chunk con un LLM.
3. Combinar resúmenes parciales y volver a resumir.
4. Post-procesar: deduplicar, asegurar tono y estilo.

**Trampas.**

- **Alucinaciones:** los resúmenes abstractive pueden
  inventar hechos. Usar RAG (recuperar antes de
  resumir) o grounding.
- **Sesgo de longitud:** los modelos tienden a generar
  resúmenes de longitud similar al training set.
  Especificar max_length.
- **Métricas malas:** ROUGE mide n-gramas, no
  semántica. Usar BERTScore o LLM-as-judge.

## Constrúyelo

```python
import numpy as np


def textrank(sentences, embeddings, damping=0.85, n_iter=100):
    """TextRank: PageRank sobre oraciones."""
    n = len(sentences)
    sim = embeddings @ embeddings.T  # (n, n)
    # Normalizar
    norm = np.linalg.norm(sim, axis=1, keepdims=True)
    sim = sim / (norm @ norm.T + 1e-12)
    sim = np.maximum(sim, 0)
    # PageRank
    M = sim / (sim.sum(axis=1, keepdims=True) + 1e-12)
    scores = np.ones(n) / n
    for _ in range(n_iter):
        scores = (1 - damping) / n + damping * M.T @ scores
    return scores


def rouge_1(candidate, reference):
    """ROUGE-1: unigramas en común / total en referencia."""
    from collections import Counter
    cand = Counter(candidate.lower().split())
    ref = Counter(reference.lower().split())
    overlap = sum((cand & ref).values())
    return overlap / max(sum(ref.values()), 1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-summarization
fase: 05
leccion: 12
---

Eres un asistente que ayuda con resumen de texto. Reci-
birás el documento y el caso de uso. Tu trabajo:

1. Si quieres SOTA y hospedado: GPT-4o, Claude 3.5
   Sonnet con prompt estructurado.
2. Si quieres open-source: PEGASUS, BART, T5.
3. Si documento es largo: dividir en chunks,
   resumir cada uno, combinar resúmenes.
4. Si necesitas factual: extractive (TextRank) o
   RAG + abstractive.
5. Prompt: "Resume el siguiente texto en 3 oraciones
   preservando los hechos clave".
6. Evaluar con ROUGE + BERTScore + LLM-as-judge.
7. Advertir contra alucinaciones: verificar hechos
   clave contra el original.
8. Post-procesar: deduplicar, verificar longitud.
```

## Ejercicios

1. **TextRank**: implementa y aplica a un artículo
   de noticias.
2. **BART**: usa HuggingFace para resumir un
   documento.
3. **Desafío**: implementa hierarchical summarization
   para documentos > 100k tokens.

## Lecturas recomendadas

- *TextRank* — Mihalcea & Tarau, 2004.
- *BART* — Lewis et al., 2020.
- *PEGASUS* — Zhang et al., 2020.
- *ROUGE* — Lin, 2004.
- HuggingFace transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Text Summarization]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
