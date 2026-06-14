# Fase 9 — Aprendizaje por refuerzo

> La base de RLHF y los agentes que juegan.

El **aprendizaje por refuerzo (RL)** es la rama de la IA donde un
agente aprende una política de decisión interactuando con un
entorno. Aunque RL no es la técnica dominante en producción hoy
día, es la base conceptual de **RLHF** (el alineamiento de los
LLMs modernos), de los **agentes que juegan** (AlphaGo, OpenAI
Five, DeepMind's AlphaStar) y de la **robótica** (manipulación,
locomoción, *sim-to-real*). Esta fase recorre el espectro desde
los MDPs formales hasta PPO y reward modeling, con la misma
filosofía del currículo: **construir antes de usar**.

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–3) sienta las bases formales: MDPs, programación
dinámica (value/policy iteration) y métodos Monte Carlo. El
**bloque 2** (4–5) entra al **model-free RL**: Q-learning, SARSA y
Deep Q-Networks. El **bloque 3** (6–8) cubre los **policy
gradients**: REINFORCE, Actor-Critic, A2C/A3C y PPO — la familia
dominante en RLHF y RLHF-like. El **bloque 4** (9–12) entra al
**RL moderno**: reward modeling y RLHF, multi-agente, *sim-to-
real* y aplicaciones a juegos.

## Índice de lecciones

### Bloque 1 — Bases formales

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [MDPs, estados, acciones y recompensas](01-mdps-estados-acciones-y-recompensas/) | Aprender | Procesos de decisión de Markov, *return*, descuento. |
| 02 | [Programación dinámica](02-programacion-dinamica/) | Construir | Value iteration, policy iteration, *Bellman optimality*. |
| 03 | [Métodos Monte Carlo](03-metodos-de-monte-carlo/) | Construir | Estimación de valor por *rollouts*, *exploring starts*. |

### Bloque 2 — Model-free RL

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 04 | [Q-Learning y SARSA](04-q-learning-y-sarsa/) | Construir | *Off-policy* vs *on-policy*, ε-greedy, convergencia. |
| 05 | [Deep Q-Networks (DQN)](05-dqn/) | Construir | *Experience replay*, *target network*, Double DQN. |

### Bloque 3 — Policy gradients

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [Policy Gradients — REINFORCE](06-policy-gradients-y-reinforce/) | Construir | *Likelihood ratio*, *baseline*, varianza. |
| 07 | [Actor-Critic: A2C, A3C](07-actor-critic-a2c-y-a3c/) | Construir | *Advantage estimation*, *asynchronous* actors. |
| 08 | [PPO](08-ppo/) | Construir | *Clipped objective*, *GAE*, implementación en PyTorch. |

### Bloque 4 — RL moderno

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 09 | [Reward modeling y RLHF](09-reward-modeling-y-rlhf/) | Construir | Modelo de preferencia, Bradley-Terry, RLHF paso a paso. |
| 10 | [Multi-agent RL](10-rl-multi-agente/) | Construir | Juegos cooperativos, *self-play*, MA-PPO. |
| 11 | [Sim-to-real transfer](11-transfer-sim-to-real/) | Construir | *Domain randomization*, *system identification*. |
| 12 | [RL para juegos](12-rl-para-juegos/) | Construir | Atari, MuJoCo, *reward shaping*. |

## Prerrequisitos

- **Fases 1, 2 y 3** completas.
- Conocimiento sólido de NumPy y PyTorch.
- Opcional: nociones de probabilidad condicional y muestreo.
- GPU recomendada para las lecciones 9–12 (RLHF, multi-agente,
  juegos).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Derivar** la ecuación de Bellman e implementarla en NumPy
  para value/policy iteration.
- **Entrenar** un agente DQN en CartPole o Atari y diagnosticar
  problemas de *reward shaping* y *exploration*.
- **Implementar** REINFORCE, A2C y PPO desde cero en PyTorch.
- **Explicar** por qué PPO es la base de RLHF.
- **Construir** un pipeline de reward modeling con preferencias
  humanas y aplicar PPO sobre el reward aprendido.
- **Comprender** los retos del multi-agente (no-estacionariedad,
 信用分配) y las técnicas de *sim-to-real*.

## Stack y herramientas

- **NumPy** y **PyTorch** para los algoritmos.
- **Gymnasium** (sucesor de Gym) para entornos.
- **Stable-Baselines3** como referencia de implementación.
- **RLlib** (Ray) para RL distribuido.
- **Tianshou** como framework alternativo.
- **PettingZoo** para multi-agente.
- **MuJoCo** y **Isaac Gym** para simulación física.
- **Weights & Biases** para tracking.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **MDP** | Lección 01 | Formalismo base. |
| **Ecuación de Bellman** | Lecciones 01, 02 | Toda la fase. |
| **Value function** | Lecciones 02–05 | Base del actor-critic. |
| **Q-learning** | Lección 04 | Algoritmo fundamental. |
| **Policy gradient** | Lección 06 | PPO, A2C, REINFORCE. |
| **PPO** | Lección 08 | RLHF (Lección 09), agentes. |
| **Reward model** | Lección 09 | Fase 10 (LLM alignment). |
| **GAE** | Lección 08 | *Advantage estimation* estable. |
| **Sim-to-real** | Lección 11 | Robótica. |

## Cómo estudiar esta fase

1. **No te saltes las lecciones 1–4.** La comprensión formal de
   MDPs y Bellman es la base de todo lo demás.
2. **CartPole es tu banco de pruebas.** Resuélvelo con Q-learning,
   DQN y PPO antes de saltar a Atari.
3. **La lección 08 (PPO) es el corazón de la fase moderna.** Si
   puedes implementarlo desde cero, entiendes el resto.
4. **RLHF (lección 09) es el puente a la Fase 10.** Conecta
   teoría de RL con el alineamiento de LLMs.
5. **Gymnasium reemplazó a Gym en 2022.** Si encuentras código
   antiguo, adáptalo a la nueva API.

## Verificación de progreso

```bash
# Lección 02 — value iteration en GridWorld
python3 fases/09-aprendizaje-por-refuerzo/02-programacion-dinamica/code/main.py

# Lección 05 — DQN en CartPole
python3 fases/09-aprendizaje-por-refuerzo/05-dqn/code/main.py

# Lección 08 — PPO en Pendulum-v1
python3 fases/09-aprendizaje-por-refuerzo/08-ppo/code/main.py
```

Si los tres agentes alcanzan el *reward* objetivo de su entorno,
la fase está aprobada.

## Cuándo usar cada algoritmo

| Problema | Algoritmo | Lección |
|---|---|---|
| Entorno pequeño, modelo conocido | Value/Policy iteration | 02 |
| Episodios cortos, modelo libre | Monte Carlo | 03 |
| Estado discreto, acción discreta | Q-learning, SARSA | 04 |
| Estado continuo (imagen) | DQN, Double DQN | 05 |
| Acción continua, política estocástica | REINFORCE, A2C, PPO | 06–08 |
| Alineamiento de LLM | RLHF (PPO + reward model) | 09 |
| Múltiples agentes | MA-PPO, self-play | 10 |
| Robots físicos | Sim-to-real + PPO | 11 |

## Conexión con otras fases

- **Entrada** → [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md).
- **Salida natural** → [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md)
  (RLHF es la base del alineamiento) y [Fase 14 — Ingeniería de
  agentes](../14-ingenieria-agentes/README.md) (los agentes usan
  técnicas de decisión secuencial).
- **Reuso en** → Fase 11 (RLHF en producción), Fase 15 (sistemas
  autónomos), Fase 16 (enjambres).

## Recursos recomendados

- *Reinforcement Learning: An Introduction* — Sutton & Barto (PDF libre).
- *Spinning Up in Deep RL* — OpenAI.
- *Stable Baselines3 docs* — <https://stable-baselines3.readthedocs.io>.
- *RLHF paper* — Christiano et al., 2017.
- *InstructGPT paper* — Ouyang et al., 2022.
- *Proximal Policy Optimization* — Schulman et al., 2017.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *MDP*,
  *Bellman*, *PPO*, *reward model*, *RLHF*.
- [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
