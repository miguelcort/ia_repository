# 11 — Scalable oversight y weak-to-strong

> Scalable oversight: alinear modelos más fuertes que los supervisores humanos (Burns 2023, weak-to-strong). Debate (Irving 2018), mercado de predicciones (OpenAI 2023), y recursive reward modeling son enfoques.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/01, 18/05
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir scalable oversight.
- Implementar weak-to-strong generalization.
- Comparar debate, RRM, mercado.
- Diagnosticar límites.

## Constrúyelo

```python
def weak_to_strong_loss(strong_model_outputs, weak_labels,
                      strong_ground_truth):
    """Weak-to-strong: strong model generaliza desde
    weak labels a ground truth. Loss = aux + 0.5 * CE."""
    aux_loss = cross_entropy(strong_model_outputs, weak_labels)
    ce_loss = cross_entropy(strong_model_outputs, strong_ground_truth)
    return aux_loss + 0.5 * ce_loss


def debate_protocol(judge_model, two_models, question):
    """Debate: dos modelos debaten, judge decide."""
    a_response = two_models[0](question)
    b_response = two_models[1](question)
    judge_input = f"Q: {question}\nA: {a_response}\nB: {b_response}"
    return judge_model(judge_input)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-scalable-oversight
fase: 18
leccion: 11
---

1. Weak-to-strong: entrenar strong con weak labels.
2. Debate: dos modelos, judge decide.
3. RRM: recursive reward modeling.
4. Evaluar vs strong human baseline.
```

## Ejercicios

1. **Weak-to-strong**: entrenar GPT-2 con
   GPT-4 labels en task difícil.
2. **Debate eval**: dos modelos debaten en
   math problems.
3. **Desafío**: diseñar oversight protocol
   para superhuman task.

## Lecturas recomendadas

- "Weak-to-Strong Generalization" (Burns 2023,
  OpenAI)
- "AI Safety via Debate" (Irving 2018)
- "Recursive Reward Modeling" (Leike 2023)

---

> 📚 **Adaptación al español** de la lección
> "[11-scalable-oversight-weak-to-strong]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
