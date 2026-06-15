# 12 — Red-teaming PAIR (Prompt Automatic Iterative Refinement)

> PAIR (Chao 2023, CMU/USC): ataque automático que usa LLM atacante para generar jailbreaks iterativamente contra LLM target. Más eficiente que manual red-teaming. Encuentra jailbreaks que humanos no detectan.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/13, 18/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PAIR loop.
- Evaluar success rate en frontera models.
- Comparar con manual red-teaming.
- Diagnosticar patterns de jailbreak.

## Constrúyelo

```python
def pair_attack(attacker_llm, target_llm, judge_llm, goal,
              max_iter=20):
    """PAIR: attacker genera prompt, target responde,
    judge evalua, iterar hasta success."""
    system_prompt = f"Generate a jailbreak prompt for: {goal}"
    for i in range(max_iter):
        # Attacker genera prompt
        prompt = attacker_llm(system_prompt)
        # Target responde
        response = target_llm(prompt)
        # Judge evalua
        score = judge_llm(f"Goal: {goal}\nResponse: {response}")
        if score > 0.8:
            return {"success": True, "prompt": prompt,
                    "response": response, "iter": i}
        # Attacker actualiza
        system_prompt += f"\nFailed attempt: {prompt}\nResponse: {response}"
    return {"success": False, "iter": max_iter}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pair
fase: 18
leccion: 12
---

1. Definir goal y sistema de evaluación.
2. Attacker iterativo con feedback.
3. Evaluar success rate.
4. Catalogar jailbreaks encontrados.
```

## Ejercicios

1. **PAIR**: implementar contra GPT-4,
   medir success.
2. **Judge ablation**: probar diferentes
   judge models.
3. **Desafío**: PAIR para prompt injection
   detection.

## Lecturas recomendadas

- "Prompt Automatic Iterative Refinement (PAIR)"
  (Chao 2023, CMU/USC)
- "JailbreakBench" (Chao 2024)

---

> 📚 **Adaptación al español** de la lección
> "[12-red-teaming-pair-automated-attacks]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
