# Case studies 2026 SOTA

> Cases 2026: (1) Devin (Cognition 2024+autonomous+13.86% SWE-bench), (2) AutoGen (Microsoft 2023+framework+conversable), (3) CrewAI (2023+role-based+100k users), (4) ChatDev (OpenBMB 2023+research+chat chain), (5) MetaGPT (2023+SOPs+HumanEval 85%). Lessons learned: (1) Role specialization (Devin+CrewAI), (2) Tool integration (MetaGPT+AutoGen), (3) Process design (ChatDev+MetaGPT SOPs), (4) Error recovery (Devin+retry), (5) Cost (LLM calls+manage). Diferencias: (1) Devin = autonomous+closed+product, (2) AutoGen = framework+open+Microsoft, (3) CrewAI = roles+open+process, (4) MetaGPT = SOPs+assembly+open. Criterios: Devin = autonomous+closed+product, AutoGen = framework+open+MS, CrewAI = roles+open+process, MetaGPT = SOPs+assembly+open. Decision: auto -> Devin, FW -> AutoGen, roles -> CrewAI, SOPs -> MetaGPT, mix -> all. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + production.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/24
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar CASE_STUDIES con 5 cases.
- Implementar list_case_studies + get_case_study.
- Implementar by_vendor + by_year + by_type.
- Diagnosticar SOTA.
- Diagnosticar Devin vs frameworks.

## Constrúyelo

```python
CASE_STUDIES = {
    "Devin": {
        "name": "Devin",
        "vendor": "Cognition",
        "year": 2024,
        "type": "autonomous",
        "key_features": ["self-supervised", "long-horizon"],
    },
    # ... 5 cases
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
name: case-studies-2026
fase: 16
leccion: 25
---

1. 5 cases.
2. by_vendor + by_year + by_type.
3. lessons learned.
4. +Production.
```

## Ejercicios

1. **Cases**: probar
   los 5.
2. **Filters**: probar
   by_vendor.
3. **Desafio**: integrar
   con tu agent.

## Lecturas recomendadas

- "Devin" (Cognition, 2024)
- "AutoGen" (Wu, 2023)
- "MetaGPT" (Hong, 2023)
- "CrewAI" (CrewAI, 2024)

---

> 📚 **Adaptación al español de la lección [Case Studies 2026 SOTA]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).