# Parallel swarm networks

> Swarm: (1) Many agents (parallel+concurrent), (2) Fan-out (distribute+1-to-N), (3) Fan-in (collect+N-to-1), (4) Aggregation (concat+first+majority), (5) Throughput (max+latency reduce). fan_out: iterate (for agent) + call (agent(task)) + return List. fan_in: concat ([r['result'] for r]), first (results[0]['result']), majority (counts+max). Swarm: agents List+reducer String+run(task) fan_out+fan_in+run_many(tasks)+history. Reducers: concat (all results+List), first (first only+fast), majority (most common+counts). Criterios: Swarm = parallel+throughput+independent, Sequential = order+dependent+pipeline, Hierarchical = levels+tree+delegation, Flat = peer+no central+simple. Decision: parallel -> swarm, order -> sequential, levels -> hierarchical, peer -> flat. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + swarm.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar fan_out + fan_in con reducers.
- Implementar Swarm con agents + reducer.
- Implementar run + run_many + history.
- Diagnosticar reducers.
- Diagnosticar swarm vs other.

## Constrúyelo

```python
class Swarm:
    def run(self, task):
        results = fan_out(self.agents, task)
        final = fan_in(results, self.reducer)
        self.history.append({"task": task, "results": results, "final": final})
        return final
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: parallel-swarm-networks
fase: 16
leccion: 09
---

1. fan_out + fan_in.
2. Swarm + reducer.
3. run + run_many.
4. +Production.
```

## Ejercicios

1. **fan_out**: probar
   con 3 agents.
2. **Swarm**: probar
   reducers.
3. **Desafio**: implementar
   async fan_out.

## Lecturas recomendadas

- "MapReduce" (Dean, 2008)
- "Parallel Computing" (Foster, 1995)
- "Anthropic: Parallel Tools" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Parallel Swarm Networks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).