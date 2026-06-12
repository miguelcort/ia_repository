# Darwin Godel machine

> Darwin Godel Machine (2025): self-improving AI via evolutionary + recursive. Modifies own code, fitness-based selection. +Open-ended, +Self-improving, +Iterative, +Evolutionary, +Recursive, +Standard, +Production. Variants: DGM seminal, Godel Agent (+recursive +code modification +standard +self-modifying), Self-modifying, custom. Variants related: AlphaEvolve (DeepMind 2025 +SOTA +evolutionary coding), Voyager (Wang 2023 +curriculum +code +Minecraft). Frameworks: custom, deepmind, openai, anthropic. +Production: standard 2024-25. +Use cases: agent, self-improving, code, evolution, recursive. Decision: open-ended -> DGM, recursive -> Godel, coding -> AlphaEvolve, curriculum -> Voyager, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + self-improving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/02, 15/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar DarwinGodelMachine con population + improvements.
- Implementar evaluate + mutate_agent + step.
- Diagnosticar DGM vs Godel Agent vs AlphaEvolve.

## Constrúyelo

```python
class DarwinGodelMachine:
    def step(self, eval_fn, mutate_fn):
        scored = self.evaluate(eval_fn)
        scored.sort(key=lambda x: x[1], reverse=True)
        new_agents = []
        for a, _ in scored[:self.population_size // 2]:
            new_agents.append(a)
        while len(new_agents) < self.population_size:
            parent = scored[len(new_agents) % len(scored)][0]
            new_agents.append(self.mutate_agent(parent, mutate_fn))
        self.agents = new_agents
        self.generation += 1
        return scored[0]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: dgm
fase: 15
leccion: 04
---

1. DGM initialize.
2. Evaluate fitness.
3. Mutate top.
4. Track improvements.
5. +Self-improving.
```

## Ejercicios

1. **DGM**: implementar
   custom DGM con population.
2. **Mutations**: probar
   diferentes mutations.
3. **Desafio**: full
   self-improving system.

## Lecturas recomendadas

- "Darwin Godel Machine: Self-Improving AI" (2025)
- "Godel Agent: Recursive Self-Improvement" (2025)
- "AlphaEvolve" (Google DeepMind, 2025)
- "Self-Modifying AI Systems" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Darwin Godel Machine]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).