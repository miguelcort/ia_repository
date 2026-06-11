# Actor-Critic A2C y A3C

> Actor-critic: actor (policy) + critic (value) comparten backbone. A2C: sync, GAE advantage, single learner. A3C: async, multiple workers, cada uno con su env. GAE (Schulman 2016): A_t = Σ(γλ)^i · δ_{t+i}, λ=0.95 típico, control bias-variance. IMPALA: distributed A2C con V-trace. Extensiones: PPO, SAC, TD3. Hoy: PPO discreto, SAC continuo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/06-policy-gradients-y-reinforce
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ActorCritic con shared backbone.
- Implementar A2C update.
- Implementar n-step y GAE advantage.
- Comparar A2C vs A3C.

## Constrúyelo

```python
class ActorCritic:
    def update(self, state, action, reward, next_state, done):
        probs, value = self.forward(state)
        _, next_value = self.forward(next_state)
        target = reward if done else reward + self.gamma * next_value
        advantage = target - value
        # Update critic y actor
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-actor-critic
fase: 09
leccion: 07
---

1. Actor pi, Critic V(s).
2. A2C sync, A3C async.
3. GAE lambda 0.95.
4. IMPALA, V-trace.
5. PPO, SAC, TD3.
```

## Ejercicios

1. **CartPole**: entrenar A2C en
   CartPole-v1.
2. **GAE**: comparar lambda 0, 0.5, 1.
3. **Desafio**: implementar IMPALA
   simplificado.

## Lecturas recomendadas

- "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (Espeholt et al., 2018)

---

> 📚 **Adaptación al español** de la lección "[Actor Critic A2C A3C]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).