# 29 — Moderation systems: OpenAI Moderation, Perspective, LlamaGuard

> Moderation APIs: OpenAI Moderation (multi-label, multi-language), Perspective API (toxicity), LlamaGuard (Meta, configurable). Production-grade content classification. F1 ~85-90% en hate speech, ~70% en casos sutiles.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/16, 18/26
**Tiempo estimado:** ~25 minutos

## Objetivos

- Conocer APIs de moderación.
- Implementar moderation pipeline.
- Comparar accuracy.
- Diagnosticar edge cases.

## Constrúyelo

```python
def openai_moderation(text, client):
    """OpenAI Moderation API. Returns multi-label scores."""
    response = client.moderations.create(input=text)
    return {
        "flagged": response.results[0].flagged,
        "categories": dict(response.results[0].categories),
        "scores": dict(response.results[0].category_scores),
    }


def perspective_toxicity(text, api_key):
    """Perspective API: TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, etc."""
    return {"TOXICITY": 0.0, "SEVERE_TOXICITY": 0.0}


def llamaguard_classify(text, policy_categories, model_id="meta/llama-guard-3-8b"):
    """LlamaGuard: classify contra custom policy."""
    chat = [{"role": "user", "content": text}]
    response = llamaguard_inference(chat, model_id=model_id)
    return response["category"]


def moderation_pipeline(text):
    """Pipeline: ensemble de moderations."""
    return {
        "openai": openai_moderation(text),
        "perspective": perspective_toxicity(text),
        "llamaguard": llamaguard_classify(text),
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
name: prompt-moderation
fase: 18
leccion: 29
---

1. OpenAI Moderation API.
2. Perspective API.
3. LlamaGuard (configurable).
4. Ensemble en production.
```

## Ejercicios

1. **OpenAI Moderation**: evaluar en
   dataset propio.
2. **LlamaGuard**: custom policy.
3. **Desafío**: ensemble voting
   pipeline.

## Lecturas recomendadas

- "OpenAI Moderation API" (2024)
- "Perspective API" (Google Jigsaw)
- "LlamaGuard 3" (Meta 2024)

---

> 📚 **Adaptación al español** de la lección
> "[29-moderation-systems-openai-perspective-llamaguard]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
