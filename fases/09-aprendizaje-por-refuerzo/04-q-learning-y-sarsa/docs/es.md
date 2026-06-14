# 04 — Diferencia temporal: Q-Learning y SARSA

> Monte Carlo espera hasta que el episodio termina. TD actualiza tras cada paso *bootstrappeando* la siguiente estimación de valor. Q-learning es *off-policy* y optimista; SARSA es *on-policy* y cauta. Ambos son una línea de código. Ambos sustentan todos los métodos de deep-RL en esta fase.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 9 · 01 (MDPs), Fase 9 · 02 (Programación dinámica), Fase 9 · 03 (Monte Carlo)
**Tiempo estimado:** ~75 minutos

## Objetivos de aprendizaje

- Derivar la actualización TD(0) y la noción de error TD.
- Implementar Q-learning y SARSA desde cero en Python puro.
- Comparar el comportamiento *off-policy* de Q-learning con el
  *on-policy* de SARSA en tareas con riesgo (cliff-walking).
- Diagnosticar las causas más comunes de no convergencia
  (α mal elegido, ε congelado, max-bias).
- Identificar la familia de deep-RL (DQN, A2C, PPO, SAC) como
  capas sobre la actualización TD de un paso.

## El problema

Monte Carlo funciona pero tiene dos demandas costosas. Necesita
episodios que terminen, y solo actualiza después de que el
*return* final está adentro. Si tu episodio es 1.000 pasos, MC
espera 1.000 pasos para actualizar algo. Es de varianza alta,
sesgo bajo, y lento en la práctica.

La programación dinámica tiene el perfil opuesto — respaldos
*bootstrapped* de varianza cero — pero requiere un modelo
conocido.

El aprendizaje por **diferencia temporal (TD)** divide la
diferencia. A partir de una sola transición `(s, a, r, s')`,
forma un objetivo de un paso `r + γ V(s')` y empuja `V(s)`
hacia él. Sin modelo. Sin episodios completos. Sesgo por usar un
`V` aproximado en el lado derecho, pero varianza dramáticamente
más baja que MC y actualizaciones *online* desde el paso uno.

Este es el pivote sobre el que gira todo el RL moderno — DQN,
A2C, PPO, SAC. El resto de la Fase 9 son capas de aproximación
de funciones y trucos construidos sobre la actualización TD de
un paso que vas a escribir en esta lección.

## El concepto

**La actualización TD(0) para V:**

```text
V(s) ← V(s) + α [r + γ V(s') - V(s)]
```

La cantidad entre corchetes es el **error TD** `δ = r + γ V(s') - V(s)`.
Es el análogo *online* de `G_t - V(s_t)` en MC. La convergencia
requiere `α` satisfaciendo Robbins-Monro (`Σ α = ∞`,
`Σ α² < ∞`) y que todos los estados se visiten infinitamente a
menudo.

**Q-learning.** Un método TD *off-policy* para control:

```text
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]
```

El `max` asume que la política *greedy* será seguida desde `s'`
en adelante, sin importar qué acción tome el agente. Ese
*desacoplamiento* hace que Q-learning aprenda `Q*` mientras el
agente explora con ε-greedy. Mnih et al. (2015) convirtieron
esto en deep Q-learning sobre Atari (lección 05).

**SARSA.** Un método TD *on-policy*:

```text
Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
```

El nombre es la tupla `(s, a, r, s', a')`. SARSA usa la acción
`a'` que el agente *realmente* toma a continuación, no el
`argmax` *greedy*. Converge a `Q^π` para cualquier `π`
ε-greedy corriendo, que en el límite `ε → 0` se vuelve `Q*`.

**La diferencia en cliff-walking.** En la tarea clásica de
cliff-walking (caer-por-el-acantilado = reward -100), Q-learning
aprende el camino óptimo a lo largo del borde del acantilado pero
ocasionalmente toma la penalización durante la exploración. SARSA
aprende un camino más seguro a un paso del acantilado porque
factoriza ruido de exploración en su Q-value. Con entrenamiento,
ambos alcanzan el óptimo a `ε → 0`. En la práctica importa:
cuando la exploración realmente está pasando en deployment, el
comportamiento de SARSA es más conservador.

**Expected SARSA.** Reemplaza `Q(s', a')` con su valor esperado
bajo `π`:

```text
Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]
```

Varianza más baja que SARSA (sin muestra de `a'`), mismo objetivo
*on-policy*. A menudo el *default* en libros de texto modernos.

**n-step TD y TD(λ).** Interpola entre TD(0) y MC esperando `n`
pasos antes de *bootstrappear*. `n=1` es TD, `n=∞` es MC.
TD(λ) promedia sobre todos los `n` con pesos geométricos
`(1-λ)λ^{n-1}`. La mayoría de deep-RL usa `n` entre 3 y 20.

## Constrúyelo

### Paso 1: SARSA con política ε-greedy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Ocho líneas. La *única* diferencia con Q-learning es la línea del
objetivo.

### Paso 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

El `max` desacopla el objetivo del comportamiento. Ese único
símbolo es la diferencia entre *on-policy* y *off-policy*.

### Paso 3: curvas de aprendizaje

Trackea el *return* medio cada 100 episodios. Q-learning
converge más rápido en GridWorld determinista simple; SARSA es
más conservadora en cliff-walking. En el 4×4 GridWorld en
`code/main.py`, ambos están cerca del óptimo tras ~2.000
episodios con `α=0.1, ε=0.1`.

### Paso 4: comparar con la verdad de DP

Corre value iteration (lección 02) para obtener `Q*`. Verifica
`max_{s,a} |Q_aprendido(s,a) - Q*(s,a)|`. Un agente TD tabular
saludable cae dentro de `~0.5` en el 4×4 GridWorld tras 10.000
episodios.

## Trampas comunes

- **Valores iniciales de Q importan.** Init optimista
  (`Q = 0` para una tarea de reward negativo) fomenta
  exploración. Init pesimista puede atrapar una política
  *greedy* para siempre.
- **Schedule de α.** α constante está bien para problemas no
  estacionarios. α que decae `α_n = 1/n` da convergencia en
  teoría pero es demasiado lento en la práctica — fíjalo en
  `[0.05, 0.3]` y monitorea la curva de aprendizaje.
- **Schedule de ε.** Empieza alto (`ε=1.0`), decae a
  `ε=0.05`. "GLIE" (*greedy in the limit with infinite
  exploration*) es la condición de convergencia.
- **Max bias en Q-learning.** El operador `max` está sesgado
  hacia arriba cuando `Q` es ruidoso. Lleva a sobreestimación
  — Double Q-learning de Hasselt (usado por DDQN en lección
  05) lo arregla con dos tablas Q.
- **Episodios no terminales.** TD puede aprender sin terminales,
  pero necesitas o capear los pasos o manejar el bootstrap
  correctamente en el cap. Estándar: trata el cap como no
  terminal, sigue *bootstrappeando*.
- **Hashing de estados.** Si los estados son tuplas/tensores,
  usa una clave *hasheable* (tupla, no lista; tupla de floats
  redondeados, no crudos).

## Úsalo

El panorama TD 2026:

| Tarea | Método | Razón |
|---|---|---|
| Entornos tabulares pequeños | Q-learning | Aprende la política óptima directamente. |
| On-policy safety-critical | SARSA / Expected SARSA | Conservadora durante la exploración. |
| Estado de alta dimensionalidad | DQN (Fase 9 · 05) | Q-función con red neuronal con replay y target net. |
| Acciones continuas | SAC / TD3 (Fase 9 · 07) | Actualización TD sobre Q-network; policy net emite acciones. |
| RL de LLM (basado en reward model) | PPO / GRPO (Fase 9 · 08, 12) | Actor-critic con ventaja estilo TD vía GAE. |
| RL offline | CQL / IQL (Fase 9 · 08) | Q-learning con regularización conservadora. |

Noventa por ciento del "RL" que lees en papers de 2026 es alguna
elaboración de Q-learning o SARSA. Entiende la actualización
tabular en tus dedos antes de leer más profundo.

## Despliégalo

Guarda como `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Elegir entre Q-learning, SARSA, Expected SARSA para una tarea tabular o de features pequeñas.
version: 1.0.0
fase: 9
leccion: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Dado un entorno tabular o de features pequeñas, devuelve:

1. Algoritmo. Q-learning / SARSA / Expected SARSA / variante
   n-step. Razón de una oración ligada a on-policy vs
   off-policy y varianza.
2. Hiperparámetros. α, γ, ε, schedule de decaimiento.
3. Inicialización. Valor Q_0 (optimista vs cero) y justificación.
4. Diagnóstico de convergencia. Curva de aprendizaje objetivo,
   check `|Q - Q*|` si DP es posible.
5. Caveat de deployment. ¿Cómo se comportará la exploración en
   inferencia? ¿Se necesita el conservadurismo de SARSA?

Rechaza aplicar TD tabular a espacios de estado > 10⁶. Rechaza
entregar un agente Q-learning sin un caveat de max-bias. Marca
cualquier agente entrenado con ε mantenido en 1.0 durante todo
el entrenamiento (sin fase de explotación).
```

## Ejercicios

1. **Fácil.** Implementa Q-learning y SARSA sobre el 4×4
   GridWorld. Grafica curvas de aprendizaje (return medio cada
   100 episodios) por 2.000 episodios. ¿Quién converge más
   rápido?
2. **Medio.** Construye un entorno de cliff-walking (4×12, la
   última fila es el acantilado con reward -100 y reset al
   inicio). Compara las políticas finales de Q-learning y
   SARSA. Captura los caminos que cada uno toma. ¿Cuál está
   más cerca del acantilado?
3. **Difícil.** Implementa Double Q-learning. Sobre un
   GridWorld con reward ruidoso (ruido Gaussiano σ=5 añadido al
   reward por paso), muestra que Q-learning sobreestima
   `V*(0,0)` por una cantidad significativa mientras Double
   Q-learning no.

## Términos clave

| Término | Lo que dice la gente | Lo que realmente significa |
|---|---|---|
| **TD error** | "La señal de actualización" | `δ = r + γ V(s') - V(s)`, el residuo *bootstrapped*. |
| **TD(0)** | "TD de un paso" | Actualización tras cada transición usando solo la estimación del siguiente estado. |
| **Q-learning** | "RL 101 off-policy" | Actualización TD con `max` sobre acciones del siguiente estado; aprende `Q*` sin importar la política de comportamiento. |
| **SARSA** | "Q-learning on-policy" | Actualización TD usando la siguiente acción real; aprende `Q^π` para la π ε-greedy actual. |
| **Expected SARSA** | "El SARSA de varianza baja" | Reemplaza la `a'` muestreada con su expectativa bajo π. |
| **GLIE** | "Schedule de exploración correcto" | Greedy in the Limit with Infinite Exploration; necesario para convergencia de Q-learning. |
| **Bootstrapping** | "Usar la estimación actual en el objetivo" | Lo que distingue TD de MC. Fuente de sesgo pero reducción masiva de varianza. |
| **Maximization bias** | "Q-learning sobreestima" | `max` sobre estimaciones ruidosas está sesgado hacia arriba; arreglado por Double Q-learning. |

## Lecturas recomendadas

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — el paper original y prueba de convergencia.
- [Sutton & Barto (2018). Cap. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — fix para maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — motivación de Expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — el paper que acuñó SARSA (entonces llamado "modified connectionist Q-learning").
- [Sutton & Barto (2018). Cap. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — generaliza TD(0) a TD(n), el camino de Q-learning a eligibility traces y, más tarde, GAE en PPO.

---

> 📚 **Adaptación al español** de la lección
> "[Temporal Difference — Q-Learning & SARSA]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
