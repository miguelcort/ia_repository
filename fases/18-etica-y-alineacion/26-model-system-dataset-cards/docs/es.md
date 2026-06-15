# 26 — Model, system, dataset cards

> Model cards (Mitchell 2019), system cards (Anthropic), dataset cards (Gebru 2021, HuggingFace): documentación estandarizada para transparencia. Incluyen intended use, training data, evals, limitations, ethical considerations.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/24
**Tiempo estimado:** ~25 minutos

## Objetivos

- Conocer model/system/dataset card format.
- Generar card automáticamente.
- Documentar limitations y risks.
- Diagnosticar transparency gaps.

## Constrúyelo

```python
def model_card(model, training_data, evals_results, intended_use,
             out_of_scope, ethical_considerations):
    """Generar model card template."""
    return {
        "model_details": {
            "name": model.name,
            "version": model.version,
            "type": model.architecture,
            "parameters": model.n_parameters,
        },
        "intended_use": intended_use,
        "training_data": {
            "datasets": training_data,
            "preprocessing": model.preprocessing,
        },
        "evaluations": evals_results,
        "limitations": model.known_limitations,
        "out_of_scope": out_of_scope,
        "ethical_considerations": ethical_considerations,
    }


def dataset_card(dataset):
    """HuggingFace dataset card format."""
    return {
        "dataset_name": dataset.name,
        "description": dataset.description,
        "language": dataset.languages,
        "size": dataset.size,
        "splits": dataset.splits,
        "annotation_process": dataset.annotation,
        "considerations": {
            "biases": dataset.known_biases,
            "limitations": dataset.limitations,
        },
    }


def system_card(agent_system, safety_evaluations):
    """Anthropic system card format."""
    return {
        "system": agent_system.description,
        "deployment": agent_system.deployment_context,
        "safety": safety_evaluations,
        "external_reviewers": agent_system.reviewers,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-model-card
fase: 18
leccion: 26
---

1. Generar model card template.
2. Documentar intended use y out-of-scope.
3. Incluir evals y limitations.
4. Publicar en HuggingFace.
```

## Ejercicios

1. **Model card**: generar para tu
   modelo.
2. **Dataset card**: HuggingFace format.
3. **Desafío**: diseñar system card
   para agent.

## Lecturas recomendadas

- "Model Cards for Model Reporting" (Mitchell 2019)
- "Datasheets for Datasets" (Gebru 2021)
- "Anthropic System Cards" (Anthropic 2024)

---

> 📚 **Adaptación al español** de la lección
> "[26-model-system-dataset-cards]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
