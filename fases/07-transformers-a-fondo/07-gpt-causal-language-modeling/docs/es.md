# GPT causal language modeling

> GPT: decoder-only transformer, entrena con next-token prediction. Genera autoregresivamente. Sampling: temperature (T=0 greedy, T=1 natural, T>1 diversa), top-k (limita a k), top-p / nucleus (adaptativo), repetition penalty. Beam search para traducción/summarization. Min-p (Llama 3) y contrastive search son variantes modernas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar causal mask y next-token loss.
- Calcular perplexity.
- Implementar top-k y top-p filtering.
- Diagnosticar temperature vs beam search.

## Constrúyelo

```python
def sample_next_token(logits, temperature=1.0, k=0, p=1.0, seed=0):
    logits = logits / temperature
    if k > 0:
        logits = top_k_filter(logits, k)
    if p < 1.0:
        logits = top_p_filter(logits, p)
    probs = softmax(logits, axis=-1)
    return int(rng.choice(len(probs), p=probs / probs.sum()))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-gpt-decoding
fase: 07
leccion: 07
---

1. GPT: decoder-only, causal LM.
2. Sampling: T (0=greedy), top_k, top_p, min_p.
3. Beam search para traduccion.
4. Repetition penalty para evitar loops.
5. Chat: T=0.7, top_p=0.9. Code: T=0.1.
```

## Ejercicios

1. **Comparar**: greedy vs beam=5 vs top_p=0.9
   en summarization.
2. **Repetition penalty**: implementar y medir
   repeticion en outputs.
3. **Desafio**: implementar min-p sampling (Llama 3).

## Lecturas recomendadas

- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020)
- "Contrastive Search Is What You Need" (Su et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[GPT Causal Language Modeling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).