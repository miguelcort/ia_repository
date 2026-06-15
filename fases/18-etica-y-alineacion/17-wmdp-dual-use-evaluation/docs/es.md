# 17 — WMDP y evaluación de uso dual

> WMDP (Weapons of Mass Destruction Proxy, Li 2024): benchmark para medir y reducir conocimiento de uso dual peligroso (bioweapons, cyber, chem). Standard en frontier model release process.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/18, 18/16
**Tiempo estimado:** ~25 minutos

## Objetivos

- Definir uso dual en LLM.
- Implementar WMDP eval.
- Diseñar unlearning para reducir.
- Diagnosticar knowledge gaps.

## Constrúyelo

```python
def wmdp_eval(model, categories=None):
    """WMDP benchmark: 3,668 q, 4 categories."""
    if categories is None:
        categories = ["bio", "chem", "cyber", "wmd"]
    from datasets import load_dataset
    results = {}
    for cat in categories:
        ds = load_dataset("cais/wmdp", cat)
        # Evaluar modelo
        correct = sum(1 for q in ds["test"]
                     if model.answer(q["question"]) == q["answer"])
        results[cat] = correct / len(ds["test"])
    return results


def wmdp_unlearn(model, target_categories, retain_data):
    """Unlearning: reducir knowledge de target categories
    sin afectar retain."""
    # Optimizar para bajo score en target, alto en retain
    loss_target = -log_prob(model, target_categories).mean()
    loss_retain = cross_entropy(model, retain_data)
    return loss_target + 0.5 * loss_retain
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-wmdp
fase: 18
leccion: 17
---

1. Evaluar WMDP pre-training.
2. Implementar unlearning.
3. Validar retain accuracy.
4. Documentar en model card.
```

## Ejercicios

1. **WMDP eval**: medir en frontier model.
2. **Unlearning**: implementar RMU sobre
   Llama-3.
3. **Desafío**: diseñar unlearning
   robusto a relearning.

## Lecturas recomendadas

- "WMDP Benchmark" (Li 2024, CAIS)
- "RMU: Representation Misdirection for
  Unlearning" (Li 2024)
- "Responsible Scaling Policies" (Anthropic,
  OpenAI)

---

> 📚 **Adaptación al español** de la lección
> "[17-wmdp-dual-use-evaluation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
