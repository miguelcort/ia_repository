# 08 — Inicialización de pesos y estabilidad

> Una inicialización correcta es la diferencia entre convergencia en minutos y horas, o divergencia total.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa,
                  04-funciones-de-activacion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar inicialización Xavier (Glorot) y He.
- Diagnosticar vanishing y exploding gradients.
- Conocer inicializaciones modernas (LSUV, Fixup).
- Entender la interacción entre inicialización y
  activación.

## El problema

Inicializas los pesos de tu red con valores aleatorios
gaussianos con desviación 1.0. Las activaciones de las
primeras capas explotan a NaN, o colapsan a 0, y la red
nunca aprende. La inicialización correcta mantiene la
varianza de las activaciones y los gradientes estable
capas abajo, lo que es prerequisito para que la red
entrene. La lección cubre Xavier, He, y el análisis
detrás de cada uno.

## El concepto

**Análisis de varianza.** Para una capa
`h^{l+1} = W^l h^l + b^l` con `W^l` de dimensión
`d_l × d_{l-1}` y asunción de inicialización simétrica
(varianza igual para cada fila de `W^l`):

- **Forward:** `Var(h^{l+1}) = d_{l-1} · Var(W^l) · Var(h^l) + Var(b^l)`.
  Si queremos `Var(h^{l+1}) = Var(h^l)`, entonces
  `Var(W^l) = 1 / d_{l-1}`.
- **Backward:** por análisis similar,
  `Var(W^l) = 1 / d_l` para mantener estable la varianza
  del gradiente.

**Xavier (Glorot) initialization.** Combina las dos
condiciones promediando geométricamente:

```text
Var(W^l) = 2 / (d_{l-1} + d_l)
```

En la práctica: `W ~ Uniform(-sqrt(6 / (d_in + d_out)),
sqrt(6 / (d_in + d_out)))` o `W ~ Normal(0, sqrt(2 /
(d_in + d_out)))`. Es la default para activaciones
saturantes (sigmoid, tanh).

**He (Kaiming) initialization.** Para ReLU y
variantes, la mitad de las neuronas "mueren" en cada
forward pass (las que reciben `x < 0`). Para compensar:

```text
Var(W^l) = 2 / d_{l-1}
```

Equivalente: `W ~ Normal(0, sqrt(2 / d_in))`. Es la
default para ReLU, Leaky ReLU, GELU. **Usa siempre He con
ReLU.**

**Tabla resumen.**

| Activación | Inicialización | Varianza |
|---|---|---|
| Sigmoid, tanh | Xavier | 2 / (d_in + d_out) |
| ReLU, Leaky ReLU, GELU | He | 2 / d_in |
| SELU | LeCun normal | 1 / d_in |
| Sin activación (lineal) | Xavier | 1 / d_in |

**Inicialización a cero.** Inicializar todos los pesos a
cero es fatal: todas las neuronas son simétricas, calculan
el mismo gradiente, y se quedan ahí para siempre. Los
**biases** sí se pueden (y a menudo se deben) inicializar
a cero.

**LSUV (Layer-Sequential Unit-Variance).** Algoritmo
iterativo: forward un mini-batch por la primera capa,
calcula la varianza de las activaciones, escala los pesos
para que la varianza sea 1. Repite capa por capa. Más
costoso que He pero converge más rápido en redes muy
profundas.

**Trampas comunes.**

- **Inicialización muy grande (var = 1.0):** activa
  todas las neuronas ReLU, gradientes saturados o
  NaN.
- **Inicialización muy pequeña (var = 1e-10):**
  activaciones colapsan a 0, gradientes también.
- **Xavier con ReLU:** subóptimo. Las neuronas ReLU
  inactivas reducen la varianza efectiva a la mitad, lo
  que sesga el análisis de Xavier.
- **No reinicializar al cambiar arquitectura:** si
  pasas de sigmoid a ReLU, también cambia de Xavier a He.

**Diagnóstico en entrenamiento.** Cómo saber si la
inicialización es correcta:

1. **Forward:** imprime la varianza de las activaciones
   de cada capa al inicio. Debe ser ~1.
2. **Backward:** imprime la norma del gradiente de cada
   capa. Debe ser comparable entre capas.
3. **Loss en los primeros pasos:** debe bajar
   rápidamente. Si se queda plana por más de 100 pasos,
   sospecha mala inicialización.

## Constrúyelo

```python
import numpy as np


def xavier_uniform(shape, gain=1.0):
    """Xavier (Glorot) uniform: U(-a, a) con a = gain *
    sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = shape[1], shape[0]
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-a, a, size=shape)


def xavier_normal(shape, gain=1.0):
    """Xavier normal: N(0, gain^2 * 2 / (fan_in + fan_out))."""
    fan_in, fan_out = shape[1], shape[0]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)


def he_uniform(shape, gain=2.0 ** 0.5):
    """He (Kaiming) uniform: U(-a, a) con a = gain * sqrt(3
    / fan_in). Para ReLU gain=sqrt(2)."""
    fan_in = shape[1]
    a = gain * np.sqrt(3.0 / fan_in)
    return np.random.uniform(-a, a, size=shape)


def he_normal(shape, gain=2.0 ** 0.5):
    """He normal: N(0, gain^2 * 2 / fan_in). Para ReLU
    gain=sqrt(2)."""
    fan_in = shape[1]
    std = gain * np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size=shape)


def lecun_normal(shape):
    """LeCun normal: N(0, 1 / fan_in). Para SELU."""
    fan_in = shape[1]
    return np.random.normal(0, np.sqrt(1.0 / fan_in), size=shape)


def zeros_bias(shape):
    return np.zeros(shape)


def ones_weight(shape):
    return np.ones(shape)


class Inicializador:
    """Helper para inicializar todas las matrices W de un MLP."""

    def __init__(self, dims, activaciones):
        """activaciones: lista de L strings."""
        self.W = []
        self.b = []
        for l, (d_in, d_out, act) in enumerate(zip(
            dims[:-1], dims[1:], activaciones
        )):
            if act in ("relu", "leaky_relu", "gelu"):
                self.W.append(he_normal((d_out, d_in)))
            elif act in ("selu",):
                self.W.append(lecun_normal((d_out, d_in)))
            else:  # sigmoid, tanh, none
                self.W.append(xavier_normal((d_out, d_in)))
            self.b.append(zeros_bias(d_out))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-inicializacion
fase: 03
leccion: 08
---

Eres un asistente que ayuda a diagnosticar problemas de
inicialización. Recibirás la arquitectura, la activación, y
los síntomas (loss no baja, gradientes 0 o NaN, etc.). Tu
trabajo:

1. Si ReLU: usar He normal (gain=sqrt(2)).
2. Si sigmoid/tanh: usar Xavier normal.
3. Si SELU: usar LeCun normal.
4. Si la loss no baja y los gradientes son ~0: varianza
   de pesos muy pequeña, aumentar inicialización.
5. Si la loss es NaN: varianza muy grande, reducir.
6. Inicializar biases a 0 siempre.
7. Si cambias de activación, también cambia de
   inicialización.
8. Para transformers, los frameworks ya manejan esto;
   confiar en el default.
```

## Ejercicios

1. **He vs Xavier**: entrena un MLP de 10 capas con ReLU y
   ambas inicializaciones; compara la curva de loss.
2. **Diagnóstico de gradientes**: implementa un check
   que imprima la norma de gradientes por capa y verifica
   que sea estable.
3. **Desafío**: implementa LSUV (Layer-Sequential
   Unit-Variance) y compara convergencia con He.

## Lecturas recomendadas

- *Understanding the difficulty of training deep feedforward
  neural networks* — Glorot & Bengio, 2010 (Xavier).
- *Delving Deep into Rectifiers* — He et al., 2015.
- *All you need is a good init* — Mishkin & Matas, 2016
  (LSUV).
- PyTorch nn.init: <https://pytorch.org/docs/stable/nn.init.html>.

---

> 📚 **Adaptación al español** de la lección "[Weight Initialization and Stability]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
