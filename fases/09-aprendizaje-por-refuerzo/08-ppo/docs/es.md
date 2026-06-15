# 08 — PPO

> PPO (Proximal Policy Optimization, Schulman et al., 2017) es el algoritmo de policy gradient más usado en RL moderno. Es estable, fácil de implementar, y se usa en RLHF, game AI, y robotics.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-policy-gradient
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Entender el problema de policy gradient: alta varianza
  y divergencia.
- Implementar la loss de PPO con clipped objective.
- Conocer las variantes: PPO-Clip, PPO-Penalty (KL),
  PPO con GAE.
- Diagnosticar cuándo usar PPO vs A2C, SAC, o DQN.

## El problema

Policy gradient (REINFORCE, A2C) tiene dos problemas:

1. **Alta varianza:** un solo batch puede dar updates
   muy grandes.
2. **Divergencia:** un update muy grande puede romper
   la política.

PPO (Schulman et al., 2017) introduce un clipped
objective que limita cuánto puede cambiar la política
por update. Es simple, estable, y se ha convertido
en el default de policy gradient. La lección cubre la
teoría y la implementación.

## El concepto

**Trust region (TRPO, Schulman et al., 2015).** Limita
KL divergence entre política vieja y nueva:

```text
max E[ π_θ(a|s) / π_θ_old(a|s) · A(s,a) ]
s.t. E[KL(π_θ_old || π_θ)] ≤ δ
```

Complejo, requiere optimización restringida.

**PPO-Clip (Schulman et al., 2017).** En vez de
constraint, clip el ratio:

```text
L^CLIP(θ) = E[ min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t) ]
```

donde `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` y
`A_t` es la advantage function.

Si la advantage es positiva (buena acción), r se
clipea a 1+ε (max 1.27x). Si es negativa (mala
acción), r se clipea a 1-ε (min 0.77x). Esto
previene updates grandes.

**Ventajas de PPO.**

- **Simple:** solo la loss clipped + entropy bonus.
- **Estable:** no diverge como policy gradient
  vanilla.
- **Eficiente:** no necesita Hessian (como TRPO).
- **General:** funciona en continuo y discreto.

**GAE (Generalized Advantage Estimation, Schulman
et al., 2016).** Estima advantage con n-step:

```text
A_t^GAE(γ,λ) = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

Combina sesgo de n-step con varianza de MC.

**Pseudocódigo PPO.**

1. Recolectar trayectorias con política actual.
2. Calcular rewards-to-go y advantages con GAE.
3. Para K epochs, minimizar -L^CLIP + c1 * (V_target - V)²
   - c2 * H(π).
4. Repetir.

**Cuándo usar PPO.**

- **Default para policy gradient:** estable y
  eficiente.
- **LLM fine-tuning (RLHF, GRPO):** PPO/GRPO para
  alinear con preferencias.
- **Game AI:** OpenAI Five, Dota 2.
- **Robotics:** MuJoCo, simulated envs.

**Cuándo NO usar PPO.**

- **Discrete action con high-dim observation:** DQN
  es competitivo.
- **Continuous control con off-policy:** SAC es más
  sample-efficient.
- **Recursos limitados:** PPO es on-policy y
  necesita muchas muestras.
- **Online learning:** DQN es más rápido.

**Trampas.**

- **Learning rate muy alto:** política diverge.
  Default 3e-4.
- **Batch size muy pequeño:** high variance. 2048-4096
  steps.
- **No normalizar advantages:** crítico para
  estabilidad.
- **Demasiados epochs:** overfitting a la
  trayectoria. Típico 3-10.

## Constrúyelo

```python
import numpy as np


def ppo_clip_loss(ratio, advantage, eps=0.2):
    """PPO clipped objective.
    ratio: π_θ / π_θ_old. advantage: A(s,a).
    Devuelve -L^CLIP (negativo porque minimizamos)."""
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * advantage
    return -np.minimum(unclipped, clipped).mean()


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation.
    rewards: (T,). values: (T+1,). dones: (T,)."""
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages
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

Eres un asistente que ayuda con policy gradient. Reci-
birás la tarea. Tu trabajo:

1. Para policy gradient en general: usar PPO con
   GAE.
2. Para RLHF: PPO o GRPO (más simple).
3. Para continuous control con muchas muestras: PPO.
4. Para sample efficiency: SAC (off-policy) en vez
   de PPO.
5. Para high-dim observation con discrete actions:
   DQN.
6. Learning rate 3e-4, batch 2048-4096, epochs 3-10.
7. Normalizar advantages siempre.
8. Advertir contra LR alto, sin normalización, y
   demasiados epochs.
```

## Ejercicios

1. **PPO en CartPole**: implementa y entrena hasta
   500 reward.
2. **PPO vs A2C**: compara estabilidad y sample
   efficiency.
3. **Desafío**: implementa PPO para Humanoid
   (MuJoCo) o Atari.

## Lecturas recomendadas

- *Proximal Policy Optimization Algorithms* — Schulman
  et al., 2017.
- *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* — Schulman et al., 2016.
- *Trust Region Policy Optimization* — Schulman et al.,
  2015.
- Spinning Up: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>.
- CleanRL PPO: <https://github.com/vwxyzjn/cleanrl>.

---

> 📚 **Adaptación al español** de la lección "[PPO]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
