# Generative agents simulation

> Gen agents: (1) Park 2023 (Stanford+Smallville), (2) Memory stream (observations+importance), (3) Reflection (synthesize+periodic), (4) Planning (goals+steps), (5) Social sim (emergent+multi-agent). GenerativeAgent: name+persona+memory_stream list+observe(event) UUID+ts+importance+_assess_importance (keywords: love/hate/death/birth/fight/friend)+reflect() recent+summary+plan_day(goals) steps+recall(query) match+sort. Ventajas generative vs role-based: memory (persistent+recall), reflection (synthesize+insight), emergent (surprising+novel), realistic (believable+human-like), social (interaction+emergent). Criterios: Generative = sim+social+emergent, Role-based = tasks+specific+cheap, Scripted = predictable+fast+test. Decision: sim -> generative, tasks -> role, predict -> scripted, mix -> generative+role. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + simulation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/16
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar GenerativeAgent con memory_stream.
- Implementar observe con importance scoring.
- Implementar reflect periodic.
- Implementar plan_day + recall.
- Diagnosticar generative vs role-based.

## Constrúyelo

```python
class GenerativeAgent:
    def observe(self, event):
        entry = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "event": event,
            "importance": self._assess_importance(event),
        }
        self.memory_stream.append(entry)
        return entry
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: generative-agents-simulation
fase: 16
leccion: 17
---

1. GenerativeAgent.
2. observe + reflect.
3. plan + recall.
4. +Production.
```

## Ejercicios

1. **GenerativeAgent**: probar
   observe + recall.
2. **reflect**: probar
   periodic.
3. **Desafio**: implementar
   un sim Smallville.

## Lecturas recomendadas

- "Generative Agents" (Park et al., 2023)
- "Smallville" (Park, 2023)
- "Memory Streams" (Park, 2023)

---

> 📚 **Adaptación al español de la lección [Generative Agents Simulation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).