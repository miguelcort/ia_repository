# MDPs: estados, acciones y recompensas

> MDP (Markov Decision Process): tupla (S, A, P, R, γ). S=estados, A=acciones, P=transiciones, R=rewards, γ=discount. Política π(a|s). Value function Vπ(s), Q-function Qπ(s,a), Advantage A(s,a) = Q - V. Ecuación de Bellman: V(s) = max_a Σ P(s'|s,a) · (R + γ·V(s')). Markov, estacionario, episodic vs continuing. Soluciones exactas: value iteration, policy iteration, LP.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-fundamentos-de-ml, 03-nucleo-de-deep-learning
**Tiempo estimado:** ~30 minutos

## Objetivos

- Calcular discounted return.
- Implementar policy evaluation iterativa.
- Implementar policy improvement greedy.
- Diagnosticar Markov y estacionariedad.

## Constrúyelo

```python
def policy_evaluation(P, R, policy, gamma, n_states, n_actions, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            a = policy[s]
            V[s] = sum(P[s, a, s2] * (R[s, a] + gamma * V[s2])
                       for s2 in range(n_states))
            delta = max(delta, abs(v - V[s]))
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
name: prompt-mdp
fase: 09
leccion: 01
---

1. MDP = (S, A, P, R, gamma).
2. V(s), Q(s,a), A(s,a).
3. Bellman equation.
4. gamma ~ 0.9-0.99.
5. Markov, episodic.
```

## Ejercicios

1. **Value iteration**: resolver MDP 4x4
   gridworld.
2. **Policy iteration**: comparar
   convergence vs VI.
3. **Desafio**: resolver Mountain Car
   con discretizacion.

## Lecturas recomendadas

- "A Markovian Decision Process" (Bellman, 1957)
- "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)
- "Dynamic Programming and Optimal Control" (Bertsekas, 2005)

---

> 📚 **Adaptación al español** de la lección "[MDPs States Actions Rewards]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).