# Q-learning y SARSA

> Q-learning (Watkins 1989): off-policy TD, Q(s,a) ← Q(s,a) + α·(r + γ·max_a' Q(s',a') - Q(s,a)). SARSA: on-policy, target = r + γ·Q(s',a') con a' real. Expected SARSA: usa expectation. Epsilon-greedy con decay. Variantes modernas: Double DQN, Dueling DQN, Distributional (C51, IQN), Rainbow. Base de DQN y RL moderno.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/03-metodos-de-monte-carlo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Q-learning update.
- Implementar SARSA update.
- Implementar epsilon-greedy.
- Diagnosticar on-policy vs off-policy.

## Constrúyelo

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma, n_actions):
    target = r + gamma * np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-q-learning
fase: 09
leccion: 04
---

1. Q-learning off-policy, max target.
2. SARSA on-policy, action real.
3. Expected SARSA.
4. Epsilon-greedy decay.
5. Double, Dueling, Rainbow.
```

## Ejercicios

1. **Q-learning**: resolver Taxi-v3 de
   Gymnasium.
2. **SARSA**: comparar vs Q-learning en
   cliff walking.
3. **Desafio**: implementar Double DQN.

## Lecturas recomendadas

- "Learning from Delayed Rewards" (Watkins, 1989)
- "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018) - Cap 6
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
- "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017)

---

> 📚 **Adaptación al español** de la lección "[Q Learning SARSA]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).