# Swarm optimization PSO ACO

> Swarm opt: (1) PSO (particles+velocity+position), (2) ACO (ants+pheromone), (3) Swarm (intelligence+collective), (4) Combinatorial (TSP+discrete), (5) Metaheuristic (heuristic+approximate). PSO: Particle (position+velocity+best_position+best_value)+iterate (for iter+for particle)+update vel=w*v+c1*r1*(best-pos)+c2*r2*(gbest-pos)+pos+=vel. ACO: pheromone Matrix+ants paths+probs+update (evaporate+deposit). PSO vs ACO: (1) PSO = continuous+gradient-free+velocities+real values+topology, (2) ACO = discrete+pheromone+paths+graph+combinatorial. Criterios: PSO = continuous+non-differentiable+swarm, ACO = discrete+combinatorial+graph, GD = smooth+differentiable+fast, LLM = logic+heuristic+discrete. Decision: continuous -> PSO, discrete -> ACO, smooth -> GD, logic -> LLM. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + swarm.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/18
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar PSO con Particle + iterate + update.
- Implementar ACO con pheromone + ants + paths.
- Diagnosticar PSO vs ACO.
- Diagnosticar criteria.
- Diagnosticar swarm opt.

## Constrúyelo

```python
def pso(objective, bounds, n_particles=10, n_iters=20, w=0.5, c1=1.5, c2=1.5, seed=None):
    # ... particle init + iterate + update
    return global_best_position, global_best_value
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: swarm-optimization-pso-aco
fase: 16
leccion: 19
---

1. PSO + ACO.
2. particle + pheromone.
3. update.
4. +Production.
```

## Ejercicios

1. **PSO**: probar
   sphere.
2. **ACO**: probar
   TSP.
3. **Desafio**: implementar
   hybrid PSO+ACO.

## Lecturas recomendadas

- "Particle Swarm Optimization" (Kennedy, 1995)
- "Ant Colony Optimization" (Dorigo, 2004)
- "Swarm Intelligence" (Bonabeau, 1999)

---

> 📚 **Adaptación al español de la lección [Swarm Optimization PSO ACO]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).