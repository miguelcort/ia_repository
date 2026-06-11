# Análisis de sentimiento

> Determinar la polaridad o emocion de un texto. Lexicon (VADER, AFINN), ML clasico (TF-IDF + LR/SVM), deep (LSTM, BERT). En produccion, BERT fine-tune con 1-10K ejemplos es el sweet spot.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-bolsa-de-palabras-y-tfidf
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar lexicon sentiment con manejo de negacion.
- Entrenar clasificador ML sobre BoW.
- Diagnosticar lexicon vs ML vs BERT.
- Evaluar con F1 macro.

## Constrúyelo

```python
class LexiconSentiment:
    def score_text(self, texto, manejo_negacion=True):
        tokens = tokenizar(texto)
        score = 0
        n = 0
        prev_neg = False
        for t in tokens:
            if t in self.lexicon:
                s = self.lexicon[t]
                if manejo_negacion and prev_neg:
                    s = -s
                score += s
                n += 1
            prev_neg = (t in {"no", "nunca"})
        return score / n if n > 0 else 0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sentiment
fase: 05
leccion: 05
---

1. Sin datos: VADER, AFINN.
2. Pocos datos: TF-IDF + LR/SVM.
3. Con GPU: BERT fine-tune (BETO).
4. Multilingual: XLM-R.
5. Aspect: PyABSA, LCF-ATEPC.
6. Few-shot: GPT-4, Claude.
```

## Ejercicios

1. **VADER en espanol**: implementar lexicon VADER-style
   en espanol.
2. **ABSA**: implementar aspect-based sentiment con
   clasificador por aspecto.
3. **Desafio**: BERT fine-tune en TASS (Taller de
   Analisis de Sentimiento en espanol), alcanzar F1 > 0.85.

## Lecturas recomendadas

- "VADER" (Hutto & Gilbert, 2014)
- "BERT Post-Training for Review Reading Comprehension and
  Aspect-based Sentiment Analysis" (Xu et al., 2019)
- PyABSA: <https://github.com/yangheng95/PyABSA>

---

> 📚 **Adaptación al español** de la lección "[Sentiment Analysis]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).