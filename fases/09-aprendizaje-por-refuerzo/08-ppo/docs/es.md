# PPO (Proximal Policy Optimization)

> PPO (Schulman 2017, OpenAI): on-policy policy gradient con clipped objective. L_CLIP = min(r·A, clip(r, 1-ε, 1+ε)·A), ε=0.2. Componentes: L_CLIP + c1·L_VF - c2·H. K=3-10 epochs, mini-batches, GAE. RLHF: PPO con reward model + KL constraint. GRPO: variant sin critic para reasoning. Variantes: PPO, PPO-penalty. SOTA en LLMs (InstructGPT, ChatGPT, DeepSeek-R1).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/07-actor-critic-a2c-y-a3c
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar clipped objective.
- Calcular GAE advantage.
- Implementar PPO loss.
- Diagnosticar hiperparámetros.

## Constrúyelo

```python
def ppo_clipped_objective(ratio, advantage, clip=0.2):
    clipped_ratio = np.clip(ratio, 1 - clip, 1 + clip)
    return np.minimum(ratio * advantage, clipped_ratio * advantage)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ppo
fase: 09
leccion: 08
---

1. L_CLIP clip objetivo.
2. eps=0.2, GAE.
3. L_CLIP + vf + entropy.
4. K=3-10 epochs.
5. GRPO sin critic.
```

## Ejercicios

1. **CartPole**: entrenar PPO en
   CartPole-v1 con Stable Baselines3.
2. **GRPO**: implementar GRPO simple
   para math reasoning.
3. **Desafio**: implementar PPO con
   KL penalty.

## Lecturas recomendadas

- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (Guo et al., 2025)
- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
- "TRPO" (Schulman et al., 2015)

---

> 📚 **Adaptación al español** de la lección "[PPO]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).