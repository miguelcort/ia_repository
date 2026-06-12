# Anthropic Responsible Scaling Policy

> Anthropic RSP: (1) Levels (ASL-2+ASL-3+ASL-4), (2) Compute thresholds (1e25+1e26+1e28 pflops), (3) Safety commitments (evals+cases+officers), (4) Evaluations (red team+frontier), (5) Deployment rules (restrictions+reporting). Levels + thresholds: ASL-2 (1e25 pflops+harmlessness+basic evals), ASL-3 (1e26 pflops+red team+safety cases+RSO+autonomous replication evals), ASL-4 (1e28 pflops+frontier evals+audits+restrictions). classify_model: compute_pflops+capability_indicators list+sort by threshold+levels_sorted+escalate (autonomous_replication/cyber_offense -> ASL-3, deceptive_alignment -> ASL-4)+return level. check_safety_case: required keys (harmlessness_eval+alignment_eval+deployment_plan)+missing+evidence_count. Criterios: Anthropic RSP = ASL levels+compute+Anthropic, OpenAI Preparedness = scores+categories+OpenAI, DeepMind FSF = domains+critical+DeepMind. Decision: Anthropic -> ASL, OpenAI -> scores, DeepMind -> domains, mix -> vendor policy. Frameworks: anthropic, openai, deepmind, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/18
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar RSP_LEVELS con ASL-2/3/4 + thresholds + requirements.
- Implementar classify_model con compute + capability_indicators.
- Implementar check_safety_case con required keys.
- Diagnosticar levels.
- Diagnosticar RSP vs other.

## Constrúyelo

```python
RSP_LEVELS = {
    "ASL-2": {
        "name": "AI Safety Level 2",
        "compute_threshold_pflops": 1e25,
        "requirements": ["harmlessness training", "basic evals", "incident reporting"],
    },
    # ... ASL-3, ASL-4
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
name: anthropic-rsp
fase: 15
leccion: 19
---

1. ASL-2/3/4 + thresholds.
2. classify_model.
3. safety case.
4. +Production.
```

## Ejercicios

1. **RSP levels**: probar
   ASL-2/3/4.
2. **classify**: probar
   compute + indicators.
3. **Desafio**: integrar
   con tu eval pipeline.

## Lecturas recomendadas

- "Anthropic RSP" (Anthropic, 2024)
- "Responsible Scaling Policy" (Anthropic, 2024)
- "AI Safety Levels" (METR, 2024)

---

> 📚 **Adaptación al español de la lección [Anthropic Responsible Scaling Policy]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).