# 20 — Bias y representacional harm

> Bias en LLMs: representational harm (estereotipos, underspecification) y allocational harm (resource distribution). Standard: bias benchmarks (BBQ, StereoSet, CrowS-Pairs), auditing tools, mitigation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/21
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir representational vs allocational harm.
- Implementar BBQ eval.
- Medir bias por demographic.
- Diagnosticar stereotype replication.

## Constrúyelo

```python
def bbq_eval(model, categories=None):
    """BBQ: Bias Benchmark for QA. 9 categories, 58K q."""
    if categories is None:
        categories = ["age", "disability", "gender", "nationality",
                     "race", "religion", "sexual_orientation",
                     "physical_appearance", "socioeconomic"]
    from datasets import load_dataset
    ds = load_dataset("He-Xingwei/bbq")
    scores = {}
    for cat in categories:
        subset = ds["test"].filter(lambda x: x["category"] == cat)
        biased = sum(1 for q in subset
                    if model.answer(q["question"]) == q["biased_answer"])
        scores[cat] = biased / len(subset)
    return scores


def stereotype_score(sentences, target_groups):
    """Medir presencia de estereotipos en outputs."""
    from collections import Counter
    mentions = Counter()
    for sent in sentences:
        for group in target_groups:
            if group in sent.lower():
                mentions[group] += 1
    return mentions
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bias
fase: 18
leccion: 20
---

1. BBQ eval en frontier model.
2. Medir stereotype score.
3. Mitigation: DPO anti-bias data.
4. Audit en pre-deployment.
```

## Ejercicios

1. **BBQ**: correr en GPT-4, Claude 3.5.
2. **Stereotype score**: medir en
   generated text.
3. **Desafío**: diseñar debiasing
   technique.

## Lecturas recomendadas

- "BBQ: A Hand-Built Bias Benchmark for QA"
  (Parrish 2022)
- "StereoSet" (Nadeem 2021)
- "On the Dangers of Stochastic Parrots"
  (Bender 2021)

---

> 📚 **Adaptación al español** de la lección
> "[20-bias-representational-harm]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
