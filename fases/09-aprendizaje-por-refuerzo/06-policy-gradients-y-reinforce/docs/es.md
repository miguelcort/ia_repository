# Policy gradients y REINFORCE

> Policy gradients (Sutton 2000): optimizan π(a|s, θ) directamente. ∇J(θ) = E[∇log π(a|s)·G_t]. REINFORCE (Williams 1992): Monte Carlo PG. Mejoras: baseline V(s), advantage A(s,a) = G - V, A2C, A3C, GAE. On-policy (PPO, A2C) vs off-policy (SAC, TD3). Aplicaciones: robotics, game AI, RLHF, GRPO para reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/03-metodos-de-monte-carlo, 09/05-dqn
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar softmax policy.
- Calcular discounted returns.
- Implementar REINFORCE update.
- Aplicar baseline para variance reduction.

## Constrúyelo

```python
def reinforce_episode(states, actions, rewards, W, b, gamma, lr):
    returns = discounted_returns(rewards, gamma)
    for t in range(len(states)):
        probs = policy_forward(states[t], W, b, n_actions=len(b))
        grad_log = -probs.copy()
        grad_log[actions[t]] += 1.0
        W += lr * returns[t] * np.outer(states[t], grad_log)
        b += lr * returns[t] * grad_log
    return W, b
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-policy-gradients
fase: 09
leccion: 06
---

1. REINFORCE: grad log pi * G_t.
2. Baseline V(s) reduce varianza.
3. Advantage A(s, a).
4. A2C, A3C, GAE.
5. PPO on, SAC/TD3 off.
```

## Ejercicios

1. **CartPole**: entrenar REINFORCE en
   CartPole-v1 de Gymnasium.
2. **Baseline**: comparar REINFORCE con
   y sin baseline.
3. **Desafio**: implementar A2C con
   GAE.

## Lecturas recomendadas

- "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (Sutton et al., 2000)
- "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (Williams, 1992)
- "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

---

> 📚 **Adaptación al español** de la lección "[Policy Gradients REINFORCE]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).