# Durable execution

> Frameworks de durable execution: (1) Temporal (Temporal 2019 +OSS +go/java/python/ts): workflows, activities, signals, queries, (2) Inngest (Inngest 2022 +SaaS+OSS +ts/python/go): steps, events, scheduled, (3) Restate (Restate 2023 +OSS +java/kotlin/ts/python/go): virtual objects, workflows, idempotency, (4) AWS Step Functions (AWS 2016 +SaaS +ASL): state machines, lambda, express, (5) DBOS (DBOS 2024 +OSS +python/ts/java/go): workflows, queues, checkpoint, (6) Cadence (Uber 2017 +OSS +go/java/python): workflows, activities, signals, (7) Durable Rules (2016 +OSS). Durable execution: state persisted (disk+DB), resume from checkpoint (crash+restart), idempotent steps (skip+replay), survive crashes (reliable+fault-tolerant), exactly-once (semantics+production). WorkflowContext mínimo: run_id (UUID+persistent), state dict (results+disk), steps_completed list (order+replay), step(name, fn) skip if name in steps_completed else run+store+append, checkpoint() return dict, restore(snap) recreate ctx. Criterios: Temporal -> OSS+Mature+polyglot, Inngest -> SaaS+OSS+TS, Restate -> OSS+Virtual objects, AWS SFN -> SaaS+AWS, DBOS -> OSS+DB-backed, Cadence -> OSS+Mature. Decision: OSS+Mature -> Temporal, TS first -> Inngest, virtual objects -> Restate, AWS -> Step Functions, DB -> DBOS, Uber-style -> Cadence. Frameworks: temporal, inngest, restate, aws, dbos, cadence. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + durable.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar DURABLE_FRAMEWORKS dict con 6+ frameworks.
- Implementar WorkflowContext con step + checkpoint + restore.
- Implementar retry_policy + is_durable_step.
- Implementar sleep_until_resumed.
- Diagnosticar frameworks.

## Constrúyelo

```python
class WorkflowContext:
    def step(self, name, fn, *args, **kwargs):
        if name in self.steps_completed:
            return self.state[name]
        result = fn(*args, **kwargs)
        self.state[name] = result
        self.steps_completed.append(name)
        return result

    def checkpoint(self):
        return {"run_id": self.run_id, "state": dict(self.state), "steps_completed": list(self.steps_completed)}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: durable-execution
fase: 15
leccion: 12
---

1. Temporal + Inngest.
2. Restate + AWS SFN.
3. DBOS + Cadence.
4. +Production.
```

## Ejercicios

1. **WorkflowContext**: probar
   idempotency.
2. **Checkpoint**: restore
   from snap.
3. **Desafio**: integrar
   con Temporal o Inngest.

## Lecturas recomendadas

- "Temporal: Durable Execution" (Temporal, 2019)
- "Inngest: Durable Functions" (Inngest, 2022)
- "Restate: Virtual Objects" (Restate, 2023)
- "DBOS: DB-Backed Workflows" (DBOS, 2024)

---

> 📚 **Adaptación al español de la lección [Durable Execution]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).