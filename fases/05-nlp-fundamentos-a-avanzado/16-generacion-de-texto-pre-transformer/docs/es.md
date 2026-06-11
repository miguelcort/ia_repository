# Generación de texto pre-transformer

> Autoregresivo: dado un contexto, generar el siguiente token. Sampling: greedy, beam, top-k, nucleus (top-p). Temperature controla diversidad. Repetition penalty y contrastive search evitan loops. Hoy los LLMs son los reyes, pero las tecnicas de decoding siguen vigentes.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15-modelado-de-temas
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar greedy decoding.
- Implementar temperature, top-k, nucleus filtering.
- Implementar beam search.
- Diagnosticar repeticion y contrastive search.

## Constrúyelo

```python
def nucleus_filter(logits, p=0.9):
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumul = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumul, p) + 1
    keep_idx = sorted_idx[:cutoff]
    new_probs = np.zeros_like(probs)
    new_probs[keep_idx] = probs[keep_idx]
    return new_probs / new_probs.sum()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-decoding
fase: 05
leccion: 16
---

1. Chat/creative: top-p=0.9, T=0.7-1.0.
2. Code/factual: T=0.2, top-p=0.95, rep_pen=1.1.
3. Summarization: beam 4.
4. Translation: beam 5.
5. Structured: Outlines, JSONformer.
6. KV cache, streaming, repetition penalty.
```

## Ejercicios

1. **Repetition penalty**: implementar penalizacion de
   tokens ya generados.
2. **Contrastive search**: implementar seleccion
   maximizando prob * similitud con contexto.
3. **Desafio**: implementar un loop de generacion con
   KV cache simulado y streaming.

## Lecturas recomendaciones

- "The Curious Case of Neural Text Degeneration" (Holtzman
  et al., 2020) — nucleus sampling
- "A Contrastive Framework for Neural Text Generation"
  (Su et al., 2022)
- "Structured Generation" (Outlines, JSONformer)

---

> 📚 **Adaptación al español** de la lección "[Pre-Transformer Text Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).