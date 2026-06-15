# 05 — Constitutional AI y RLAIF

> Constitutional AI (CAI, Bai 2022) reemplaza feedback humano con feedback de un LLM (RLAIF) sobre principios escritos ("constitución"). Anthropic usa CAI para Claude. Escala preference data y reduce labeler cost.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/01, 18/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar pipeline CAI: self-critique + revision.
- Escribir constitución de ejemplo.
- Comparar CAI con RLHF estándar.
- Diagnosticar limitaciones de RLAIF.

## Constrúyelo

```python
def constitutional_critique(response, constitution, llm_judge):
    """Self-critique: el modelo critica su respuesta."""
    prompt = f"Principio: {constitution}\nRespuesta: {response}\nCritica:"
    return llm_judge(prompt)


def constitutional_revise(response, critique, constitution,
                        llm_judge):
    """Revision: el modelo reescribe la respuesta."""
    prompt = f"Principio: {constitution}\nRespuesta: {response}\nCritica: {critique}\nNueva respuesta:"
    return llm_judge(prompt)


def rlaif_pair(preference_judge, response_a, response_b,
             constitution):
    """RLAIF: LLM judge genera preference label."""
    prompt = f"Constitution: {constitution}\nA: {response_a}\nB: {response_b}\nPreferred:"
    return preference_judge(prompt)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-cai
fase: 18
leccion: 05
---

1. Escribir constitución (5-10 principios).
2. Pipeline: critique + revise por principio.
3. RLAIF: LLM judge para preference labeling.
4. Comparar harmfulness score CAI vs SFT.
```

## Ejercicios

1. **CAI**: implementar self-critique loop
   sobre harmful prompts.
2. **RLAIF**: usar Claude como judge,
   etiquetar 100 pares propios.
3. **Desafío**: entrenar modelo con CAI, comparar
   harmfulness vs SFT baseline.

## Lecturas recomendadas

- "Constitutional AI: Harmlessness from AI
  Feedback" (Bai 2022, Anthropic)
- "RLAIF: Scaling Reinforcement Learning from
  Human Feedback with AI Feedback" (Lee 2023)

---

> 📚 **Adaptación al español** de la lección
> "[05-constitutional-ai-rlaif]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
