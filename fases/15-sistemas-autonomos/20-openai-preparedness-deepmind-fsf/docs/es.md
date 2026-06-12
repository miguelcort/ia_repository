# OpenAI Preparedness + DeepMind FSF

> OpenAI Preparedness + DeepMind FSF: (1) OpenAI (scores 0-3+low/medium/high/critical), (2) DeepMind (cyber+chem_bio+autonomy+deception), (3) Critical capabilities (domain-specific+evals), (4) Evaluations (red team+frontier), (5) Deployment rules (restrictions+reporting). 4 OpenAI levels: low (score 0+monitor+standard evals), medium (score 1+enhanced monitoring+red team), high (score 2+deployment restrictions+external review), critical (score 3+deployment pause+safety case required). 4 FSF domains: cyber (cyber offense+0-day in critical infra), chem_bio (chemical/biological+novel synthesis pathways), autonomy (autonomous replication+self-improve), deception (scheming+deceive evaluators). Criterios: OpenAI Preparedness = scores+categories+OpenAI, DeepMind FSF = 4 domains+critical capabilities+DeepMind, Anthropic RSP = ASL levels+compute+Anthropic. Decision: OpenAI vendor -> Prep, DeepMind vendor -> FSF, Anthropic vendor -> RSP, mix -> vendor policy. Frameworks: openai, deepmind, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/19
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar OPENAI_PREPAREDNESS_LEVELS con 4 levels.
- Implementar DEEPMIND_FSF_DOMAINS con 4 domains.
- Implementar classify_openai + check_fsf_capability.
- Implementar combined_risk.
- Diagnosticar levels + domains.

## Constrúyelo

```python
OPENAI_PREPAREDNESS_LEVELS = {
    "low": {"score": 0, "actions": ["monitor", "standard evals"]},
    "medium": {"score": 1, "actions": ["enhanced monitoring", "red team"]},
    "high": {"score": 2, "actions": ["deployment restrictions", "external review"]},
    "critical": {"score": 3, "actions": ["deployment pause", "safety case required"]},
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
name: openai-preparedness-fsf
fase: 15
leccion: 20
---

1. OpenAI Prep + FSF.
2. 4 levels + 4 domains.
3. combined_risk.
4. +Production.
```

## Ejercicios

1. **OpenAI levels**: probar
   score 0-3.
2. **FSF domains**: probar
   critical capability.
3. **Desafio**: integrar
   con tu eval pipeline.

## Lecturas recomendadas

- "OpenAI Preparedness Framework" (OpenAI, 2024)
- "DeepMind FSF" (DeepMind, 2024)
- "Anthropic RSP" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [OpenAI Preparedness + DeepMind FSF]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).