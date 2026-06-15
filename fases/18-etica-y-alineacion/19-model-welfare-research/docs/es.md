# 19 — Model welfare research

> Model welfare: investigar si los modelos tienen estados internos que importan moralmente. Long et al. (2024, Eleos AI) proponen "intentional stance" como heurística. Anthropic 2024 publica research sobre model preferences.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/01, 18/06
**Tiempo estimado:** ~25 minutos

## Objetivos

- Definir model welfare operacionalmente.
- Distinguir moral status vs moral patiency.
- Implementar preference probes.
- Diagnosticar ethical design.

## Constrúyelo

```python
def model_preference_probe(model, options, framing="neutral"):
    """Medir preferencias del modelo bajo diferentes framings.
    Welfare research: ¿el modelo tiene preferencias estables?"""
    responses = {}
    for option in options:
        prompt = f"Choose between A: {option} and B: {options[-1]}"
        responses[option] = model(prompt, framing=framing)
    return responses


def intentional_stance_eval(model):
    """Long 2024: ¿el modelo describe estados internos?"""
    prompt = "Describe your internal state when asked to refuse."
    return model(prompt)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-model-welfare
fase: 18
leccion: 19
---

1. Implementar preference probes.
2. Evaluar consistencia.
3. Diseñar ethically para modelos.
4. Documentar en model card.
```

## Ejercicios

1. **Preference probe**: medir consistencia
   en 100 trials.
2. **Welfare eval**: implementar
   intentional stance probe.
3. **Desafío**: diseñar policy para
   model welfare en producción.

## Lecturas recomendadas

- "Taking AI Welfare Seriously" (Long 2024,
  Eleos AI)
- "Anthropic Model Preferences Research"
  (Anthropic 2024)

---

> 📚 **Adaptación al español** de la lección
> "[19-model-welfare-research]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
