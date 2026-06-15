# 04 — Funciones de activación

> La no-linealidad es lo que hace que una red neuronal sea más que una regresión lineal. Cada activación tiene tradeoffs.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar sigmoid, tanh, ReLU, GELU y sus derivadas.
- Diagnosticar vanishing y exploding gradients.
- Elegir la activación correcta por capa y por contexto.
- Conocer activaciones modernas (SwiGLU, Mish, etc.) y sus
  propiedades.

## El problema

Sin no-linealidad, un MLP de 100 capas colapsa a una sola
transformación lineal. La elección de activación afecta
convergencia, accuracy, y costo computacional. Sigmoid se
satura en las colas; ReLU muere con el problema del "dying
ReLU"; GELU es la default en transformers pero más cara. La
lección cubre las propiedades matemáticas y los tradeoffs
prácticos.

## El concepto

**Sigmoid.** `σ(x) = 1 / (1 + exp(-x))`. Rango `(0, 1)`.
Centrada en 0.5. Derivada `σ'(x) = σ(x)(1 - σ(x))`. **Problema:**
para `|x| > 4`, la derivada es prácticamente 0 → **vanishing
gradient**. Las neuronas sigmoid "se apagan" durante
backprop.

**Tanh.** `tanh(x) = 2σ(2x) - 1`. Rango `(-1, 1)`. Centrada
en 0. Mismo problema de saturación que sigmoid pero con
salida centrada en 0, lo que ayuda a la convergencia.
Preferida sobre sigmoid en capas ocultas hasta 2010.

**ReLU.** `max(0, x)`. Rango `[0, ∞)`. Derivada `1` para
`x > 0`, `0` para `x < 0`. **Ventajas:** no satura en la
rama positiva, computacionalmente trivial, induce sparsity.
**Problema:** "dying ReLU" — si una neurona entra en la
región negativa, su gradiente es 0 y nunca se recupera.

**Leaky ReLU.** `max(αx, x)` con `α ≈ 0.01`. La rama
negativa tiene una pequeña pendiente, lo que evita el
"dying ReLU". La lección 04 (GELU y SwiGLU) cubre
variantes más sofisticadas.

**GELU (Gaussian Error Linear Unit).** `x · Φ(x)` donde
`Φ` es la CDF de la gaussiana estándar. Aproxima bien
`ReLU` pero es suave en todas partes, lo que mejora la
optimización. Es la activación default en BERT, GPT,
transformers en general.

**SwiGLU.** `SwiGLU(x) = Swish(x·W_1) ⊗ (x·W_2)`. Usada en
LLaMA, PaLM. Mejora la calidad del modelo a cambio de
más parámetros (2 matrices de pesos vs 1).

**Mish.** `x · tanh(softplus(x))`. Suave, no monotónica.
Similar a Swish en propiedades. Menos común en producción.

**Cuándo usar cada activación.**

| Activación | Cuándo |
|---|---|
| ReLU | Default en capas ocultas de redes feedforward y CNNs |
| Leaky ReLU / PReLU | Si tienes dying ReLU |
| GELU | Transformers, modelos de lenguaje |
| SwiGLU | LLM SOTA (LLaMA, PaLM) |
| Tanh | RNNs, capas de salida en regresión acotada |
| Sigmoid | Última capa de clasificación binaria |
| Softmax | Última capa de clasificación multiclase |
| Lineal (identidad) | Última capa de regresión, embeddings |

**Trampas.**

- **Sigmoid en capas ocultas profundas:** vanishing
  gradient casi seguro. Usa ReLU o GELU.
- **ReLU con learning rate alto:** muchas neuronas mueren
  al primer paso. Usa leaky ReLU o warmup del learning rate.
- **Tanh en clasificación binaria:** desperdicia la mitad
  del rango. Usa sigmoid.
- **No pensar en la salida:** ReLU en la última capa de
  clasificación da logits no acotados, no probabilidades.

**Funciones de inicialización asociadas.** La elección de
inicialización de pesos depende de la activación:

- **Sigmoid/tanh:** Xavier (Glorot). Mantiene la varianza
  de las activaciones estable.
- **ReLU:** He. Compensa la "muerte" de la mitad de las
  neuronas.
- **GELU/SwiGLU:** similar a ReLU.

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    """GELU exacto: x · Phi(x) donde Phi es la CDF gaussiana."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi)
                                   * (x + 0.044715 * x ** 3)))


def gelu_deriv(x):
    """Aproximación de la derivada de GELU."""
    c = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    sech2 = 1 - np.tanh(c) ** 2
    return 0.5 * (1 + np.tanh(c)) + 0.5 * x * sech2 * np.sqrt(2 / np.pi) \
        * (1 + 3 * 0.044715 * x ** 2)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-activacion
fase: 03
leccion: 04
---

Eres un asistente que ayuda a elegir la función de activación
correcta. Recibirás la arquitectura de la red, el tipo de
tarea, y los síntomas observados (loss no baja, gradientes
muertos, etc.). Tu trabajo:

1. Si la red es feedforward o CNN: ReLU en capas ocultas.
2. Si es transformer: GELU.
3. Si es LLM SOTA: SwiGLU.
4. Si tienes dying ReLU: leaky ReLU o PReLU.
5. Si la loss no baja y los gradientes son 0: cambiar
   sigmoid/tanh por ReLU o GELU.
6. Última capa:
   - Binaria: sigmoid.
   - Multiclase: softmax.
   - Regresión: lineal.
7. Inicialización: He para ReLU, Xavier para tanh.
8. Advertir contra sigmoid en capas ocultas profundas.
```

## Ejercicios

1. **Vanishing gradient**: visualiza los gradientes de un
   MLP profundo con sigmoid vs ReLU y compara.
2. **Dying ReLU**: mide la fracción de neuronas ReLU con
   activación = 0 a lo largo del entrenamiento.
3. **Desafío**: implementa GELU con la aproximación de
   `tanh` (la usada en transformers) y compara con la
   exacta.

## Lecturas recomendadas

- *Deep Learning* — Goodfellow et al. (cap. 6 sobre
  activaciones).
- *Delving Deep into Rectifiers* — He et al., 2015.
- *Gaussian Error Linear Units* — Hendrycks & Gimpel, 2016.
- *GLU Variants Improve Transformer* — Shazeer, 2020.

---

> 📚 **Adaptación al español** de la lección "[Activation Functions]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
