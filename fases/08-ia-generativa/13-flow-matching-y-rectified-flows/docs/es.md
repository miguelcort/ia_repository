# 13 — Flow matching y rectified flows

> Flow matching (Lipman et al., 2023) y rectified flows (Liu et al., 2022) son alternativas a los modelos de difusión con trayectorias de probabilidad más simples (lineales u óptimas). Ofrecen entrenamiento más estable y muestreo más rápido.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-difusion-ddpm-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender continuous normalizing flows y el teorema
  de cambio de variables.
- Implementar flow matching y rectified flows.
- Conocer las ventajas vs DDPM: training estable,
  sampling rápido.
- Diagnosticar cuándo usar flow matching vs difusión
  vs GANs.

## El problema

Los modelos de difusión (DDPM, score-based SDEs) usan
trayectorias curvas y estocásticas. Flow matching
(Lipman et al., 2023) y rectified flows (Liu et al.,
2022) usan trayectorias más simples (lineales, óptimas)
y permiten training más estable, sampling con menos
steps, y trayectorias más rectas que son ideales para
destilación. La lección cubre la teoría y la
implementación.

## El concepto

**Continuous normalizing flows (CNFs).**

Una ODE define la transformación:

```text
dx/dt = v_θ(x, t)
```

Con x(0) = x_0 y x(1) = x_1. La densidad se transforma
según el teorema de cambio de variables:

```text
log p_1(x_1) = log p_0(x_0) - ∫_0^1 div(v_θ) dt
```

Training requiere computar la divergencia, lo cual es
caro. Solución: flow matching.

**Flow matching (Lipman et al., 2023).** Condición en
un path específico. El más simple: path lineal:

```text
x_t = (1 - t) x_0 + t x_1
dx_t/dt = x_1 - x_0
```

La red aprende a predecir el vector `u_θ(x_t, t) = x_1 -
x_0`. La loss es:

```text
L = E[ || u_θ(x_t, t) - (x_1 - x_0) ||² ]
```

Esto es tan simple como la loss de DDPM, pero con
trayectorias lineales.

**Rectified flows (Liu et al., 2022).** Similar a flow
matching, pero con la observación de que las
trayectorias lineales son óptimas para transporte de
masa. Reflow (refit con el modelo mismo) reduce
curvatura. Reflow + destilación da modelos de 1-2
steps.

**Ventajas vs DDPM.**

- **Training estable:** no hay SDEs estocásticos.
- **Sampling rápido:** trayectorias rectas → menos
  steps Euler (10-20 vs 1000).
- **Destilación efectiva:** rectified flow + consistency
  model da 1-step generation.
- **Mejor para inpainting y editing:** ODE determinista
  vs SDE.

**Aplicaciones.**

- Stable Diffusion 3 (Esser et al., 2024) usa
  flow matching.
- Imagen 3, Sora, Veo usan rectified flows.
- Consistency models (Song et al., 2023) están
  relacionados.

**Cuándo usar flow matching.**

- **Imágenes:** reemplazar DDPM/DDIM con flow
  matching para training más estable.
- **1-4 step generation:** rectified flow + consistency
  model.
- **ODE-based editing:** inpainting, interpolation.
- **Video:** Sora, Veo.

**Cuándo NO usar.**

- **Aplicaciones existentes:** DDPM y score-based
  tienen un ecosistema más grande.
- **Mode coverage perfecto:** aún no demostrado que
  flow matching lo supere.
- **Sin GPU:** flow matching es tan caro como DDPM.

**Trampas.**

- **No usar linear path:** path lineal es la elección
  más simple. Otras (cosine, polynomial) son posibles
  pero menos estables.
- **Sin time embedding:** la ODE no sabe en qué
  timestep está. Usar sinusoidal o FiLM.
- **No reflow:** sin reflow, las trayectorias se curvan
  y se necesitan más steps.

## Constrúyelo

```python
import numpy as np


def linear_path(x_0, x_1, t):
    """Trayectoria lineal: x_t = (1 - t) x_0 + t x_1."""
    return (1 - t) * x_0 + t * x_1


def target_velocity(x_0, x_1):
    """Velocity objetivo: x_1 - x_0."""
    return x_1 - x_0


def flow_matching_loss(velocity_pred, x_0, x_1, t):
    """Loss de flow matching."""
    return np.mean((velocity_pred - target_velocity(x_0, x_1)) ** 2)


def euler_sample(x_0, velocity_fn, T=20):
    """Euler ODE solver: x(0) -> x(1)."""
    dt = 1.0 / T
    x = x_0
    for i in range(T):
        t = i * dt
        x = x + dt * velocity_fn(x, t)
    return x
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-flow-matching
fase: 08
leccion: 13
---

Eres un asistente que ayuda con modelos generativos.
Recibirás la tarea. Tu trabajo:

1. Para imágenes: flow matching o rectified flow son
   más estables que DDPM.
2. Para 1-step generation: rectified flow + consistency
   model.
3. Para SD 3+: usa flow matching (DiT-MM).
4. Para video: rectified flow (Sora, Veo).
5. Para ODE-based editing: flow matching es
   determinista.
6. Para evaluar: FID, IS, FVD (video).
7. Advertir contra paths no lineales y reflow
   faltante.
```

## Ejercicios

1. **Mini-flow-matching**: implementa y entrena en
   MNIST.
2. **Rectified flow**: implementa reflow y compara
   trayectorias.
3. **Desafío**: entrena un modelo de flow matching
   condicional en CIFAR-10.

## Lecturas recomendadas

- *Flow Matching for Generative Modeling* — Lipman et
  al., 2023.
- *Flow Straight and Fast: Learning to Generate and
  Transfer Data with Rectified Flow* — Liu et al., 2022.
- *Consistency Models* — Song et al., 2023.
- *Scaling Rectified Flow Transformers for High-
  Resolution Image Synthesis* — Esser et al., 2024.

---

> 📚 **Adaptación al español** de la lección "[Flow Matching and Rectified Flows]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
