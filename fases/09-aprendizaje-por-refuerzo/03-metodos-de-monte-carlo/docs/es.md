# 03 — Métodos de Monte Carlo

> Los métodos de Monte Carlo estiman expectativas sobre distribuciones usando muestreo. Son fundamentales en RL para estimar valores y políticas: REINFORCE, Monte Carlo Tree Search (MCTS), y value estimation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-bandas-y-procesos-de-decision
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el método de Monte Carlo para estimar
  expectativas.
- Implementar REINFORCE con policy gradient.
- Conocer Monte Carlo Tree Search (MCTS) y su uso en
  AlphaZero.
- Diagnosticar cuándo usar Monte Carlo vs métodos
  bootstrapping (TD, n-step).

## El problema

Los métodos de Monte Carlo (MC) estiman expectativas
sobre distribuciones usando muestreo aleatorio. En
reinforcement learning, MC se usa para:

- **Policy gradient (REINFORCE):** estimar ∇J(θ)
  muestreando trayectorias.
- **Value estimation:** estimar V(s) promediando retornos
  muestreados.
- **MCTS (Monte Carlo Tree Search):** búsqueda en árbol
  con simulaciones aleatorias. Usado en AlphaGo,
  AlphaZero, MuZero.

La lección cubre estos métodos y cuándo usar cada uno.

## El concepto

**Estimación Monte Carlo.** Para estimar E[f(X)] con
X ~ p, muestrear x_1, ..., x_N i.i.d. y calcular:

```text
Ê[f(X)] = (1/N) ∑ f(x_i)
```

Por la ley de los grandes números, Ê → E conforme
N → ∞. Varianza decrece como 1/N. Para reducir
varianza: control variates, importance sampling,
antithetic sampling.

**REINFORCE (Williams, 1992).** Policy gradient
directo:

```text
∇J(θ) = E[ ∇_θ log π_θ(a|s) · R(τ) ]
```

donde τ = (s_0, a_0, r_0, ..., s_T) es la trayectoria
y R(τ) = ∑ r_t. Implementación:

1. Muestrear trayectorias con la política actual.
2. Para cada paso, calcular el retorno R_t = ∑_{k≥t} r_k.
3. Actualizar θ: θ ← θ + α ∇_θ log π_θ(a_t|s_t) · R_t.

Problema: alta varianza. Solución: baseline (sustracción
de V(s) o una red de valor).

**Monte Carlo Tree Search (MCTS).** Búsqueda en árbol
donde cada nodo es un estado, y cada edge una acción.
Cuatro pasos:

1. **Selection:** desde la raíz, seleccionar child que
   maximice UCB (Upper Confidence Bound):
   `Q(s,a) + c · sqrt(ln N(s) / (1 + N(s,a)))`.
2. **Expansion:** añadir un nuevo child al árbol.
3. **Simulation:** rollout aleatorio desde el nuevo
   nodo hasta terminal.
4. **Backpropagation:** actualizar Q, N.

Iterar K veces. MCTS + policy network = AlphaZero
(Silver et al., 2017). MCTS + value network =
AlphaGo Zero.

**MC vs TD (Temporal Difference).**

- **MC:** actualiza al final del episodio. Sin bias,
  alta varianza. Necesita episodios finitos.
- **TD(0):** actualiza en cada step. Más bias, menos
  varianza. Online.
- **TD(λ):** compromise con eligibility traces.

**Cuándo usar Monte Carlo.**

- **Episodios finitos:** MC se adapta naturalmente.
- **Estimación de política:** REINFORCE es simple y
  efectivo.
- **Búsqueda en juegos:** MCTS es estándar.
- **No hay modelo del environment:** MC solo necesita
  rollout.

**Cuándo NO usar.**

- **Episodios largos / continuos:** MC tiene varianza
  muy alta. Usar TD o n-step.
- **Recompensas sparse:** MC puede no ver señal.
  Usar shaped rewards o curiosity-driven exploration.
- **High-dim state:** MCTS explota con dimensiones
  grandes. Usar aprendida model-based methods.

**Trampas.**

- **Alta varianza de REINFORCE:** siempre usar
  baseline. Una red de valor separada ayuda.
- **MCTS con budget bajo:** MCTS necesita suficientes
  simulaciones. < 100 = pobre.
- **Monte Carlo sin burn-in:** las primeras
  simulaciones pueden estar sesgadas. Ignorar o
  pre-train.

## Constrúyelo

```python
import numpy as np


def monte_carlo_estimate(f, sampler, N=10000):
    """Estima E[f(X)] con N muestras."""
    samples = np.array([f(sampler()) for _ in range(N)])
    return samples.mean(), samples.std() / np.sqrt(N)


def reinforce_update(policy_logits, actions, returns, lr=0.01):
    """REINFORCE update: ∇ log π(a|s) * R.
    policy_logits: (T, |A|). actions: (T,). returns: (T,)."""
    T = len(actions)
    probs = np.exp(policy_logits) / np.exp(policy_logits).sum(
        axis=-1, keepdims=True)
    log_probs = np.log(probs[np.arange(T), actions] + 1e-8)
    # Normalizar returns (reduce variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    gradient = np.zeros_like(policy_logits)
    for t in range(T):
        gradient[t, actions[t]] += returns[t] * (1 - probs[t, actions[t]])
        for a in range(probs.shape[1]):
            if a != actions[t]:
                gradient[t, a] -= returns[t] * probs[t, a]
    policy_logits += lr * gradient
    return policy_logits
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-monte-carlo
fase: 09
leccion: 03
---

Eres un asistente que ayuda con métodos de Monte Carlo
en RL. Recibirás la tarea. Tu trabajo:

1. Para policy gradient: REINFORCE con baseline.
2. Para episodios largos: usar TD(λ) o n-step en vez
   de MC puro.
3. Para juegos: MCTS + policy network (AlphaZero).
4. Para model-based: Dyna, MCTS, o learned world
   model.
5. Para high-variance: usar baseline, GAE, o
   trust region (TRPO/PPO).
6. Advertir contra MC sin baseline, episodios muy
   largos, y rewards sparse.
```

## Ejercicios

1. **Monte Carlo integration**: estima ∫ sin(x) dx
   entre 0 y π.
2. **REINFORCE**: implementa en CartPole-v1.
3. **Desafío**: implementa MCTS para Tic-Tac-Toe.

## Lecturas recomendadas

- *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning (REINFORCE)* —
  Williams, 1992.
- *Mastering the Game of Go without Human Knowledge
  (AlphaZero)* — Silver et al., 2017.
- *Reinforcement Learning: An Introduction* — Sutton &
  Barto, 2018. Cap 5 (MC).
- Lilian Weng's blog: <https://lilianweng.github.io/posts/2018-02-19-rl-overview/>.

---

> 📚 **Adaptación al español** de la lección "[Monte Carlo Methods]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
