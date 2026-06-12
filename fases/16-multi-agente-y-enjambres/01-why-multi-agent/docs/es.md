# Why multi-agent

> Por que multi-agent: (1) Parallel (multiple paths+research+code review), (2) Specialization (different roles+software team+debate), (3) Robustness (one fails+system continues+production+critical), (4) Collective (wisdom of crowds+voting+ensemble), (5) Diverse evaluation (multiple perspectives+review+safety). Downsides: (1) Coordination (latency+complexity), (2) Conflict (disagreement+voting+arbitration), (3) Cost (n*single+tokens), (4) Debugging (harder to trace+observability). Decision: (1) task_complexity 0-1 (<0.3 -> too simple), (2) parallel_value 0-1 (<0.2 -> low), (3) latency_budget 0-1 (<0.1 -> tight), (4) thresholds (complexity+parallel>0.8 -> high, else -> balanced). Criterios: Multi = complex+parallel+specialized, Single = simple+fast+cheap, Hybrid = simple core+multi for steps. Decision: complex -> multi, simple -> single, specific -> hybrid. Frameworks: langchain, anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + multi-agent.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Ninguno
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar REASONS con 5 reasons.
- Implementar DOWNSIDES con 4 downsides.
- Implementar decide_use_multi_agent con thresholds.
- Diagnosticar reasons vs downsides.
- Diagnosticar multi vs single vs hybrid.

## Constrúyelo

```python
REASONS = {
    "parallel": {
        "name": "Parallel exploration",
        "use_cases": ["research", "code review", "brainstorming"],
        "cost": "linear in agents",
    },
    # ... 5 reasons
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
name: why-multi-agent
fase: 16
leccion: 01
---

1. 5 reasons + 4 downsides.
2. decide_use_multi_agent.
3. thresholds.
4. +Production.
```

## Ejercicios

1. **Reasons**: probar
   los 5 reasons.
2. **decide**: probar
   thresholds.
3. **Desafio**: integrar
   con CrewAI o LangGraph.

## Lecturas recomendadas

- "Multi-Agent Systems" (Wooldridge, 2009)
- "CrewAI: Multi-Agent" (CrewAI, 2024)
- "AutoGen" (Microsoft, 2024)

---

> 📚 **Adaptación al español de la lección [Why Multi-Agent]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).