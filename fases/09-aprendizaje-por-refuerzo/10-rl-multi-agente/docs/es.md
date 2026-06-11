# RL multi-agente (MARL)

> MARL: múltiples agentes en stochastic games. Categorías: cooperative (shared reward, QMIX/MADDPG/MAPPO), competitive (zero-sum, self-play AlphaStar/OpenAI Five), mixed (general-sum, Nash Q-learning). Nash equilibrium: best response, PPAD-complete, self-play converge en zero-sum. CTDE: Centralised Training Decentralised Execution. SOTA: MADDPG, QMIX, MAPPO. Benchmarks: SMAC, Hanabi, Diplomacy.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/08-ppo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar stochastic game step.
- Encontrar Nash equilibrium en 2-player games.
- Implementar Q-learning MARL.
- Diagnosticar cooperative vs competitive.

## Constrúyelo

```python
def nash_equilibrium_2p_payoff(payoff_a, payoff_b):
    # Best response: a* = argmax_a payoff_a[:, b*], b* = argmax_b payoff_b[a*, :]
    equilibria = []
    for a in range(payoff_a.shape[0]):
        for b in range(payoff_a.shape[1]):
            if a == np.argmax(payoff_a[:, b]) and b == np.argmax(payoff_b[a, :]):
                equilibria.append((a, b))
    return equilibria
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-marl
fase: 09
leccion: 10
---

1. MARL: stochastic game.
2. Coop, comp, mixed.
3. Nash: best response.
4. CTDE: train central.
5. MADDPG, QMIX, MAPPO.
```

## Ejercicios

1. **Prisoner's Dilemma**: entrenar
   Q-learning contra self-play.
2. **MADDPG**: implementar MADDPG
   en PettingZoo.
3. **Desafio**: QMIX para SMAC.

## Lecturas recomendadas

- "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., 2017)
- "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (Rashid et al., 2018)
- "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., 2022)
- "Game Theory" (Myerson, 1997)

---

> 📚 **Adaptación al español** de la lección "[Multi-Agent RL]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).