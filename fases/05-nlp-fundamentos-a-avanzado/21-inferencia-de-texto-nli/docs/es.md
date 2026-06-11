# Inferencia de texto (NLI)

> Premise -> Hypothesis: entailment, neutral, contradiction. SOTA: DeBERTa-v3-large (91%+ MNLI). Aplicaciones: zero-shot classification, factuality check, retrieval reranking. Multilingual: XNLI (15 idiomas).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 20-salidas-estructuradas-y-decoding-constrenido
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar NLI 3-way mock.
- Implementar cross-entropy loss.
- Calcular accuracy.
- Diagnosticar cuando usar NLI.

## Constrúyelo

```python
def nli_3way_mock(premisa, hipotesis):
    p_words = set(premisa.lower().split())
    h_words = set(hipotesis.lower().split())
    if h_words.issubset(p_words): return 0
    if p_words & h_words: return 1
    return 2
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nli
fase: 05
leccion: 21
---

1. Zero-shot clasif: DeBERTa-MNLI, T5-MNLI.
2. Custom: DeBERTa fine-tune 1-10K pares.
3. Factuality: claim + evidencia -> entailment/contradiction.
4. Multilingual: XLM-R + XNLI.
5. Adversarial: ANLI.
```

## Ejercicios

1. **NLI BERT fine-tune**: cargar DeBERTa-MNLI y
   evaluar en SNLI.
2. **Zero-shot clasif**: implementar clasificador
   binario con NLI.
3. **Desafio**: factuality checker para RAG con
   DeBERTa-MNLI, evaluar en FEVER.

## Lecturas recomendadas

- "SNLI" (Bowman et al., 2015)
- "MultiNLI" (Williams et al., 2018)
- "DeBERTa-v3" (He et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Natural Language Inference]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).