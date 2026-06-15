# 21 — Fairness: group, individual, counterfactual

> Fairness criteria: group fairness (demographic parity, equalized odds), individual fairness (similar individuals → similar outcomes), counterfactual fairness (decision in counterfactual world). Chouldechova 2017, Kusner 2017.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir group/individual/counterfactual fairness.
- Implementar demographic parity check.
- Comparar criterios (impossibility theorem).
- Diagnosticar trade-offs.

## Constrúyelo

```python
import numpy as np


def demographic_parity(y_pred, group):
    """P(y_pred=1 | group=A) = P(y_pred=1 | group=B)."""
    groups = set(group)
    rates = {g: np.mean([p for p, gr in zip(y_pred, group) if gr == g])
            for g in groups}
    return max(rates.values()) - min(rates.values())


def equalized_odds(y_pred, y_true, group):
    """TPR y FPR iguales por grupo."""
    groups = set(group)
    metrics = {}
    for g in groups:
        mask = [gr == g for gr in group]
        tp = sum(p == 1 and t == 1 for p, t, m in zip(y_pred, y_true, mask) if m)
        fp = sum(p == 1 and t == 0 for p, t, m in zip(y_pred, y_true, mask) if m)
        pos = sum(t == 1 and m for t, m in zip(y_true, mask))
        neg = sum(t == 0 and m for t, m in zip(y_true, mask))
        metrics[g] = {"tpr": tp / max(pos, 1),
                     "fpr": fp / max(neg, 1)}
    return metrics


def counterfactual_fairness(model, x, protected_attr,
                          counterfactual_value):
    """Misma predicción si protected_attr fuera diferente."""
    original = model.predict(x)
    x_cf = x.copy()
    x_cf[protected_attr] = counterfactual_value
    cf = model.predict(x_cf)
    return original == cf
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-fairness
fase: 18
leccion: 21
---

1. Group fairness: demographic parity.
2. Individual fairness: similar → similar.
3. Counterfactual fairness: protected
   attribute no afecta.
4. Trade-off: impossibility theorem.
```

## Ejercicios

1. **Demographic parity**: implementar y
   medir en COMPAS dataset.
2. **Equalized odds**: comparar TPR/FPR
   por grupo.
3. **Desafío**: diseñar fair classifier
   con post-processing.

## Lecturas recomendadas

- "Fair Prediction with Disparate Impact"
  (Chouldechova 2017)
- "Counterfactual Fairness" (Kusner 2017)
- "A Survey on Bias and Fairness in ML"
  (Mehrabi 2021)

---

> 📚 **Adaptación al español** de la lección
> "[21-fairness-criteria-group-individual-counterfactual]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
