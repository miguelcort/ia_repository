# Resumen de texto

> Resumir un documento en version mas corta preservando info clave. ROUGE (n-gram overlap) es la metrica standard. BART, T5, PEGASUS son los SOTA encoder-decoder. LLMs (GPT-4, Claude) zero-shot con prompt son production-ready.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-traduccion-automatica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar ROUGE-N (n-gram overlap).
- Implementar ROUGE-L (longest common subsequence).
- Implementar extractivo lead-N como baseline.
- Diagnosticar extractivo vs abstractive.

## Constrúyelo

```python
def rouge_n(reference, candidate, n=1):
    ref_ngrams = Counter(n_grams(reference, n))
    cand_ngrams = Counter(n_grams(candidate, n))
    overlap = sum(min(c, ref_ngrams[k]) for k, c in cand_ngrams.items())
    return overlap / sum(ref_ngrams.values())
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

1. News: BART, T5.
2. Dialogos: SAMSum, BART.
3. Custom: BART + LoRA 1-10K pares.
4. Zero-shot: GPT-4, Claude.
5. Long context: LongT5, LED, chunking.
6. ROUGE + factuality + human eval.
```

## Ejercicios

1. **TextRank**: implementar summarization extractivo con
   grafo + PageRank.
2. **BERTScore**: usar bert-score para evaluacion
   semantica.
3. **Desafio**: BART fine-tune en CNN/DailyMail subset,
   alcanzar ROUGE-2 > 0.18.

## Lecturas recomendadas

- "ROUGE" (Lin, 2004)
- "BART" (Lewis et al., 2020)
- "PEGASUS" (Zhang et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[Text Summarization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).