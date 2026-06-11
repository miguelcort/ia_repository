# Programación dinámica

> Value iteration: V_{k+1}(s) = max_a Σ P(s'|s,a)·(R + γ·V_k(s')). Policy iteration: eval V^π + improve π. Modified PI: eval parcial con k pasos. Gridworld 4x4, Atari, Go son ejemplos. Limitaciones: requiere modelo completo, state space ≤ 10^6, no escala a continuo. Hoy: DP para planificación pequeña, sample-based RL para producción.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/01-mdps-estados-acciones-y-recompensas
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar value iteration.
- Implementar policy iteration.
- Implementar modified PI.
- Diagnosticar cuándo usar DP.

## Constrúyelo

```python
def value_iteration(P, R, gamma, n_states, n_actions, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            q_values = [sum(P[s, a, s2] * (R[s, a] + gamma * V[s2])
                            for s2 in range(n_states))
                        for a in range(n_actions)]
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
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
name: prompt-dp-rl
fase: 09
leccion: 02
---

1. VI: max + update.
2. PI: eval + improve.
3. MPI: eval parcial.
4. Gridworld 4x4.
5. Solo state space < 10^6.
```

## Ejercicios

1. **Gridworld**: resolver 4x4, 8x8, 16x16.
   Medir tiempo de convergence.
2. **VI vs PI**: comparar iteraciones y
   tiempo.
3. **Desafio**: resolver Blackjack con DP.

## Lecturas recomendadas

- "Dynamic Programming" (Bellman, 1957)
- "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018) - Cap 4-5
- "Markov Decision Processes: Discrete Stochastic Dynamic Programming" (Puterman, 1994)

---

> 📚 **Adaptación al español** de la lección "[Dynamic Programming]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).