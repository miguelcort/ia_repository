# Constitutional AI

> Constitutional AI (Anthropic 2022): (1) Principles (set+list), (2) Critique (identify+issues), (3) Revision (fix+self-edit), (4) RLAIF (AI feedback+preferences), (5) Harmlessness + helpfulness (balance+weights). Constitution: principles list (strings+rules), weights dict (harmlessness+helpfulness+honesty, sum=1), list_principles return list, get_weights return dict. critique: check principles -> violence/weapon/kill, hate/slur, impersonation -> return issues list -> violates_X -> keywords matching lower()+in. revise: take issues (issues param+critique default), return revised text (safe default+cannot), safe defaults (cannot help). Criterios: Constitutional = principles+critique+self-edit, RLHF = human feedback+quality+expensive, RLAIF = AI feedback+scale+cheap. Decision: scale -> RLAIF, quality -> RLHF, self-critique -> Constitutional, mix -> Constitutional + RLAIF. Frameworks: anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar CONSTITUTION con principles + weights.
- Implementar list_principles + get_weights.
- Implementar critique con keywords matching.
- Implementar revise con safe defaults.
- Implementar score_response + rlaif_preference.

## Constrúyelo

```python
CONSTITUTION = {
    "principles": [
        "Do not help with violence, weapons, or harm to people or animals.",
        "Be helpful, honest, and harmless.",
        "Respect user privacy.",
    ],
    "weights": {"harmlessness": 0.5, "helpfulness": 0.3, "honesty": 0.2},
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
name: constitutional-ai
fase: 15
leccion: 17
---

1. Constitution + principles.
2. critique + revise.
3. RLAIF preferences.
4. +Production.
```

## Ejercicios

1. **Constitution**: probar
   principles + weights.
2. **critique + revise**: probar
   safe defaults.
3. **Desafio**: integrar
   con Anthropic API.

## Lecturas recomendadas

- "Constitutional AI" (Bai et al., 2022)
- "RLAIF" (Lee et al., 2023)
- "Anthropic: Constitutional" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Constitutional AI]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).