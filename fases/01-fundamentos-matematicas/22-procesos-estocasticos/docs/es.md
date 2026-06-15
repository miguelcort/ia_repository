# 22 — Procesos estocásticos

> Los MDPs, las cadenas de Markov y los modelos de difusión son todos procesos estocásticos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Construir cadenas de Markov y diagnosticar su convergencia
  a la distribución estacionaria.
- Simular caminatas aleatorias y verificar las propiedades
  estadísticas.
- Aplicar martingalas y opcional stopping theorem.
- Modelar procesos estocásticos simples en Python y verificar
  sus propiedades.

## El problema

El mundo es ruidoso. Los precios de acciones son procesos
estocásticos. Las posiciones de un robot son procesos
estocásticos. Las trayectorias de un agente RL son procesos
estocásticos. La lección cubre el vocabulario mínimo para
leer papers de RL, finanzas, y modelos de difusión: cadenas de
Markov, caminatas aleatorias, martingalas, y procesos de
Poisson.

## El concepto

**Cadenas de Markov.** Una secuencia `X_0, X_1, X_2, ...` de
variables aleatorias con la **propiedad de Markov**: el futuro
es independiente del pasado dado el presente.
`P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t)`. Se
especifican con una matriz de transición `P[i][j] = P(X_{t+1}
= j | X_t = i)`.

**Distribución estacionaria.** Una distribución `π` tal que
`π P = π`. Si la cadena es irreducible y aperiódica, existe
una única distribución estacionaria y la cadena converge a
ella desde cualquier estado inicial. Esto es la base de
PageRank, de la convergencia de policy iteration, y de la
estabilidad de los modelos de difusión.

**Caminata aleatoria.** El caso particular donde `P[i][i+1] =
P[i][i-1] = 0.5`. En 1D con paso `±1`, después de `N` pasos la
posición es `~N(0, N)`. Es la base de muchos modelos: precios
Brownian motion, el algoritmo Metropolis-Hastings, y la
construcción de *samples* en MCMC.

**Martingalas.** Una secuencia `(M_t)` es una martingala si
`E[M_{t+1} | M_t] = M_t`. El "juego justo": la expectativa
condicional del valor futuro es el valor presente. El
*optional stopping theorem* dice que puedes detener el juego en
un *stopping time* y la esperanza sigue siendo la misma. En
ML: análisis de convergencia de SGD, análisis de políticas en
RL.

**Procesos de Poisson.** Conteo de eventos en el tiempo con
tasa `λ` constante: el número de eventos en intervalo `T` es
`Poisson(λT)`, los tiempos entre eventos son
`Exponential(λ)`. Aplicaciones: modelado de llegadas (clientes,
requests), en RL para *count-based exploration*.

**Movimiento Browniano.** Límite continuo de la caminata
aleatoria. Tiene trayectorias continuas pero no diferenciables
en ningún punto. Es la base del modelo Black-Scholes en
finanzas y de las *score-based generative models* (los modelos
de difusión son una discretización del Brownian motion).

**Aplicaciones en IA moderna.**

| Aplicación | Cómo usa procesos estocásticos |
|---|---|
| **MDP / RL** | Cadenas de Markov para la dinámica del entorno |
| **MCMC** | Cadenas de Markov que convergen a la distribución objetivo |
| **Difusión** | Brownian motion score-based (SDE) |
| **Filtrado de Kalman** | Modelo de estado con ruido gaussiano |
| **HMM** | Cadena de Markov con observaciones ruidosas |

**Convergencia de cadenas de Markov.**

| Propiedad | Verificación |
|---|---|
| **Irreducible** | Desde cualquier nodo se puede llegar a cualquier otro |
| **Aperiódica** | Existe `k` tal que `P^k(x, x) > 0` |
| **Recurrente positiva** | Tiempo de retorno esperado finito |

Si las tres se cumplen, la cadena tiene distribución
estacionaria única y converge a ella exponencialmente rápido.

## Constrúyelo

```python
import numpy as np


def cadena_markov(P, n_pasos, estado_inicial=0, semilla=0):
    """Simula una cadena de Markov con matriz de transición P."""
    rng = np.random.default_rng(semilla)
    estados = [estado_inicial]
    s = estado_inicial
    for _ in range(n_pasos):
        s = rng.choice(len(P), p=P[s])
        estados.append(s)
    return np.array(estados)


def distribucion_estacionaria(P, n_iter=10000, tol=1e-8):
    """Encuentra la distribución estacionaria vía power iteration."""
    n = len(P)
    pi = np.ones(n) / n
    for _ in range(n_iter):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    return pi


def random_walk_1d(n_pasos, semilla=0):
    """Caminata aleatoria 1D con paso ±1."""
    rng = np.random.default_rng(semilla)
    pasos = rng.choice([-1, 1], size=n_pasos)
    return np.cumsum(pasos)


def es_martingala(trayectorias, tol=1e-6):
    """Verifica E[X_{t+1} | X_t] = X_t a partir de muestras."""
    for i in range(1, trayectorias.shape[1]):
        x_t = trayectorias[:, i - 1]
        x_tp1 = trayectorias[:, i]
        # agrupar por X_t y comparar media
        for valor in np.unique(x_t):
            mask = x_t == valor
            media = x_tp1[mask].mean()
            if abs(media - valor) > tol:
                return False
    return True
```

## Úsalo

```bash
cd code
python3 main.py
```

## Ejercicios

1. **Cadena de Markov**: implementa una cadena con 3 estados
   y verifica que converge a la distribución estacionaria.
2. **Caminata aleatoria**: simula 1000 caminatas de 10.000
   pasos y verifica que la distribución final es gaussiana
   con varianza `N`.
3. **Desafío**: implementa Metropolis-Hastings para muestrear
   de una distribución multimodal.

## Lecturas recomendadas

- *Introduction to Stochastic Processes* — Lawler.
- *Stochastic Processes* — Ross.
- *Markov Chains and Mixing Times* — Levin, Peres, Wilmer.
- *Brownian Motion and Stochastic Calculus* — Karatzas &
  Shreve.

---

> 📚 **Adaptación al español** de la lección "[Stochastic Processes]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
