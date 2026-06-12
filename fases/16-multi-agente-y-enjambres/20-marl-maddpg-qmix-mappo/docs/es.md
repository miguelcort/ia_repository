# MARL MADDPG QMIX MAPPO

> MARL: (1) MADDPG (actor-critic+CTDE), (2) QMIX (mixing network+cooperative), (3) MAPPO (PPO multi+on-policy), (4) Cooperative (shared reward+team), (5) Competitive (zero-sum+adversarial). MultiAgentEnv: n_agents+n_actions+reset() random state+step(actions) rewards+observation(agent_id). MADDPGAgent: q_values list+select_action(obs, epsilon) epsilon-greedy+update(obs, action, reward) Q-learning. QMIXMixer: n_agents+weights list+mix(individual_qs) weighted sum. Diferencias: (1) MADDPG = off-policy+actor-critic+CTDE, (2) QMIX = cooperative+mixing+centralized, (3) MAPPO = on-policy+PPO+stable. Criterios: MADDPG = continuous+AC+off-policy, QMIX = cooperative+mixing+centralized, MAPPO = stable+on-policy+sample efficient, LLM = logic+discrete+heuristic. Decision: cont -> MADDPG, coop -> QMIX, stable -> MAPPO, logic -> LLM. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + MARL.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/19
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar MultiAgentEnv con reset + step.
- Implementar MADDPGAgent con q_values + select + update.
- Implementar QMIXMixer con weights + mix.
- Implementar maddpg_train + mappo_train.
- Diagnosticar MARL algorithms.
- Diagnosticar criteria.

## Constrúyelo

```python
class MADDPGAgent:
    def select_action(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return max(range(self.n_actions), key=lambda a: self.q_values[a])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: marl-maddpg-qmix-mappo
fase: 16
leccion: 20
---

1. MultiAgentEnv.
2. MADDPGAgent + QMIX.
3. maddpg + mappo.
4. +Production.
```

## Ejercicios

1. **MultiAgentEnv**: probar
   step.
2. **MADDPG**: probar
   update.
3. **Desafio**: implementar
   QMIX training.

## Lecturas recomendadas

- "MADDPG" (Lowe, 2017)
- "QMIX" (Rashid, 2018)
- "MAPPO" (Yu, 2022)

---

> 📚 **Adaptación al español de la lección [MARL MADDPG QMIX MAPPO]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).