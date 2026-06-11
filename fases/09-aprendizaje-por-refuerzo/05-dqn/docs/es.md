# DQN (Deep Q-Network)

> DQN (Mnih 2015, DeepMind): Q-network convolutional, experience replay buffer (1M), target network (frozen, soft/hard update), loss MSE entre Q(s,a) y r + γ·max Q_target(s',a'). Atari: 49 juegos, supera human-level. Variantes: Double DQN, Dueling DQN, Prioritized Replay, Rainbow, IQN. Limitaciones: sample-inefficient, solo acciones discretas. Base de RL moderno con value-based.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/04-q-learning-y-sarsa, 03-nucleo-deep-learning
**Tiempo estimado:** ~35 minutos

## Objetivos

- Implementar DQN con replay buffer.
- Implementar target network update.
- Implementar epsilon decay.
- Calcular prioritized experience weights.

## Constrúyelo

```python
class DQN:
    def train_step(self, batch_size=32, seed=0):
        batch = self.sample_batch(batch_size, seed=seed)
        for (s, a, r, s_next, done) in batch:
            target = r + (0 if done else self.gamma * np.max(self.target_forward(s_next)))
            q_pred = self.forward(np.asarray(s, dtype=float))[a]
            # gradient update
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dqn
fase: 09
leccion: 05
---

1. Q-network + replay + target.
2. MSE loss.
3. Soft update target.
4. Epsilon decay.
5. Rainbow = 6 mejoras.
```

## Ejercicios

1. **Atari**: entrenar DQN en
   Pong o Breakout.
2. **Double DQN**: descomponer Q
   para reducir overestimation.
3. **Desafio**: implementar Rainbow
   completo.

## Lecturas recomendadas

- "Human-level Control through Deep Reinforcement Learning" (Mnih et al., 2015)
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
- "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017)
- "Distributional Reinforcement Learning with Quantile Regression" (Dabney et al., 2018)

---

> 📚 **Adaptación al español** de la lección "[DQN]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).