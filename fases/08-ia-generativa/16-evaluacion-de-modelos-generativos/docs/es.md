# Evaluación de modelos generativos

> Benchmarks: FID, CLIP score, ImageReward, HPSv2, PickScore, GenEval, T2I-CompBench, HEIM (12 axes). Human eval: MOS, A/B preference, Elo (Chatbot Arena style). VLM-as-judge: GPT-4V, LLaVA. Pipeline riguroso: prompts diversos, multiple metrics, statistical significance, reportar limitations. No single metric es suficiente; multi-metric + human eval es gold standard.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/14-evaluacion-fid-y-clip-score
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Elo rating update.
- Calcular MOS y confidence intervals.
- Implementar binomial test para preferences.
- Diagnosticar HEIM y T2I-CompBench.

## Constrúyelo

```python
def elo_rating(rating_a, rating_b, score_a, k=32):
    exp_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-eval-generative
fase: 08
leccion: 16
---

1. Multi-metric: FID + CLIP + ImageReward.
2. Human eval: MOS, A/B, Elo.
3. HEIM: 12 axes holistic.
4. T2I-CompBench: compositionality.
5. VLM-as-judge: GPT-4V.
```

## Ejercicios

1. **HEIM**: implementar mini-HEIM
   eval con 3 axes.
2. **A/B test**: hacer human eval
   simple y statistical test.
3. **Desafio**: implementar VLM-as-judge
   pipeline con GPT-4V.

## Lecturas recomendadas

- "HEIM: Holistic Evaluation of Image Models" (Lee et al., 2023)
- "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-Image Generation" (Huang et al., 2023)
- "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Models" (Ghosh et al., 2023)
- "Chatbot Arena: Open Platform for Evaluating LLMs by Human Preference" (Chiang et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Evaluation of Generative Models]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).