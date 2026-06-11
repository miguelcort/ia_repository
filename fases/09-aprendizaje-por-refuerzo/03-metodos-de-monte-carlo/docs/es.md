# Métodos de Monte Carlo

> Monte Carlo RL: V(s) = E[Σ γ^t r_t] muestreado de episodios completos. First-visit vs every-visit MC. On-policy (ε-greedy) y off-policy (importance sampling). Bias 0, variance alta vs TD biased low-variance. Extensiones modernas: MCTS (AlphaZero, MuZero), V-trace (IMPALA), Retrace(λ), model-based rollouts (DreamerV3).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/02-programacion-dinamica
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar first-visit y every-visit MC.
- Samplear returns de episodios.
- Implementar MC control on-policy.
- Diagnosticar on vs off-policy.

## Constrúyelo

```python
def first_visit_mc(states, returns, n_states):
    V = np.zeros(n_states)
    counts = np.zeros(n_states)
    for episode_states, G in zip(states, returns):
        visited = set()
        for s in episode_states:
            if s not in visited:
                visited.add(s)
                counts[s] += 1
                V[s] += (G - V[s]) / counts[s]
    return V
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mc-rl
fase: 09
leccion: 03
---

1. MC: V(s) = mean(returns).
2. First vs every-visit.
3. On vs off-policy.
4. Bias 0, variance alta.
5. MCTS: tree + MC rollouts.
```

## Ejercicios

1. **Blackjack**: resolver con MC.
2. **Off-policy**: implementar
   importance sampling.
3. **Desafio**: MCTS para tic-tac-toe.

## Lecturas recomendadas

- "Monte Carlo Methods in RL" (Sutton & Barto, 2018) - Cap 5
- "MCTS for Game Playing" (Browne et al., 2012)
- "Mastering the Game of Go with Deep Neural Networks and Tree Search" (Silver et al., 2016)

---

> 📚 **Adaptación al español** de la lección "[Monte Carlo Methods]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).