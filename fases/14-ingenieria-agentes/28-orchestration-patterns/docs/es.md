# Orchestration patterns

> Orchestration patterns: (1) manager-worker (manager decompose + distribute subtasks + workers execute + collect results, +scalable +parallel +independent +aggregate), (2) pipeline (sequential stages, +composition +dependency +order), (3) scatter-gather (parallel workers + gather results, +speed +independent +concurrent), (4) event-driven (async +reactive +scalable), (5) map-reduce, (6) hierarchical. +Multi-agent, +Scalable, +Production, +Reliable, +Standard, +Parallel, +Efficient, +Composition, +Speed, +Concurrent. Variants: manager-worker, pipeline, scatter-gather, event-driven, map-reduce, hierarchical, custom. Frameworks: langchain, langgraph, autogen, crewai, temporal, airflow, dagster, prefect. +Production: standard 2024-25. +Use cases: agent, multi-agent, ETL, automation. Decision: decompose -> manager-worker, sequential -> pipeline, parallel -> scatter-gather, async -> event-driven, production -> combinacion. Trade-offs: cada uno + specialty, pipeline + composition, scatter-gather + speed. 2025: +MCP + A2A + native + orchestration.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13, 14/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Orchestrator con workers dict.
- Implementar manager_worker.
- Implementar pipeline.
- Implementar scatter_gather.
- Diagnosticar orchestration trade-offs.

## Constrúyelo

```python
class Orchestrator:
    def manager_worker(self, task, n_workers=3):
        subtasks = [f"subtask_{i}_{task}" for i in range(n_workers)]
        results = []
        for sub in subtasks:
            worker_name = list(self.workers.keys())[hash(sub) % len(self.workers)]
            results.append(self.workers[worker_name](sub))
        return results
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: orchestration
fase: 14
leccion: 28
---

1. Manager-worker.
2. Pipeline.
3. Scatter-gather.
4. Event-driven.
5. +Multi-agent.
```

## Ejercicios

1. **Manager-worker**:
   implementar orchestrator
   custom.
2. **Pipeline**: probar
   pipeline pattern.
3. **Desafio**: full
   orchestration system.

## Lecturas recomendadas

- "Patterns of Distributed Systems" (Unmesh Joshi, 2024)
- "Temporal: Workflow as Code" (Temporal, 2024)
- "LangGraph Orchestration" (LangChain, 2024)
- "Multi-Agent Orchestration" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Orchestration Patterns]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).