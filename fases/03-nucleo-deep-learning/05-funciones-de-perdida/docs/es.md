# 05 — Funciones de pérdida

> La pérdida convierte "qué tan mal lo hace el modelo" en un número que podemos minimizar. Cada tarea tiene su pérdida canónica.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-funciones-de-activacion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar MSE, MAE, Huber para regresión.
- Implementar cross-entropy binaria y categórica.
- Diagnosticar cuándo usar cada pérdida según la tarea.
- Conocer pérdidas avanzadas: contrastiva, focal, dice.

## El problema

Tienes un modelo que predice valores y clases. Necesitas una
función `L(ŷ, y)` que mida qué tan equivocadas están las
predicciones, y que sea **diferenciable** (para backprop). La
elección de pérdida importa tanto como la del modelo: MSE
penaliza outliers cuadráticamente, MAE es robusta, cross-
entropy es lo correcto para clasificación, focal loss
aborda desbalance extremo. La lección cubre las pérdidas
canónicas y cuándo elegir cada una.

## El concepto

**MSE (Mean Squared Error).** `L = (1/n) Σ (ŷ_i - y_i)²`. La
pérdida por defecto en regresión. Penaliza errores grandes
cuadráticamente, lo que es bueno cuando los errores
siguen una distribución normal. Derivada simple: `2(ŷ -
y)`. Sensible a outliers (un error de 10 contribute 100
veces más que uno de 1).

**MAE (Mean Absolute Error).** `L = (1/n) Σ |ŷ_i - y_i|`.
Robusta a outliers: el error contribution es lineal.
Derivada es `sign(ŷ - y)`, no suave en 0 (lo que puede
causar problemas con gradientes cerca del óptimo). Útil
cuando los outliers son errores genuinos que no quieres
dominen la pérdida.

**Huber loss.** Combina MSE y MAE: cuadrática para errores
pequeños (donde MSE es bueno) y lineal para errores
grandes (donde MAE es robusto). El threshold de transición
es `δ`:

```text
L = (1/2)(ŷ - y)² si |ŷ - y| <= δ
L = δ(|ŷ - y| - δ/2) en otro caso
```

Es la default en muchas librerías de RL.

**Cross-entropy binaria (BCE).** Para clasificación
binaria:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

donde `p = σ(z)`. Es la divergencia KL entre la distribución
verdadera y la predicha. Su gradiente combinado con
sigmoide es `p - y`, que no satura como el de MSE +
sigmoid. Esta es la razón por la que BCE es preferible a
MSE para clasificación.

**Cross-entropy categórica.** Para clasificación
multiclase con K clases:

```text
L = -Σ_k y_k log(p_k)
```

donde `p = softmax(z)`. Es la elección canónica. Tiene la
misma propiedad de gradiente limpio cuando se combina
con softmax: `p - y` por neurona de salida.

**Focal loss.** Para clasificación con desbalance
extremo (e.g. detección de objetos donde el 99.9% de las
regiones son fondo):

```text
L = -α (1 - p)^γ log(p)
```

El factor `(1 - p)^γ` down-weighta ejemplos fáciles y
enfoca el gradiente en los difíciles. `γ = 2` es el
default. Usado en RetinaNet y muchos detectores modernos.

**Contrastive loss.** Para aprender embeddings: ejemplos
similares deben tener distancia pequeña, ejemplos
diferentes deben tener distancia grande. Usado en
Siamese networks, contrastive pre-training, metric
learning.

**Cuándo usar cada pérdida.**

| Tarea | Pérdida |
|---|---|
| Regresión, errores normales | MSE |
| Regresión con outliers | MAE o Huber |
| Clasificación binaria | BCE |
| Clasificación multiclase | Cross-entropy categórica |
| Detección de objetos | Focal loss |
| Embeddings / similarity | Contrastive / triplet |
| Segmentación | Dice loss |
| Reinforcement learning | Policy gradient loss |

**Trampas.**

- **MSE en clasificación:** la saturación del gradiente
  retrasa el aprendizaje. Usa cross-entropy.
- **Cross-entropy sin `softmax` en la última capa:**
  produce NaN. Aplica softmax primero.
- **MAE sin warmup:** gradientes erráticos al inicio. Empieza
  con MSE y cambia a MAE después de algunas epochs.
- **Focal con `γ` muy alto:** ignora ejemplos fáciles
  completamente. Típico `γ = 2`.

## Constrúyelo

```python
import numpy as np


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_grad(y_true, y_pred):
    return np.sign(y_pred - y_true) / len(y_true)


def huber(y_true, y_pred, delta=1.0):
    err = np.abs(y_pred - y_true)
    return float(np.mean(
        np.where(err <= delta, 0.5 * err ** 2,
                 delta * (err - 0.5 * delta))
    ))


def bce(y_true, p_pred, eps=1e-12):
    """Cross-entropy binaria. y_true en {0, 1}, p_pred en (0, 1)."""
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred)
    ))


def bce_grad(y_true, p_pred, eps=1e-12):
    """Gradiente asumiendo p_pred = sigmoid(z). El resultado es
    simplemente p - y, sin saturación."""
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return (p_pred - y_true) / len(y_true)


def softmax(z, eps=1e-12):
    z = z - z.max(axis=-1, keepdims=True)  # estabilidad
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cce(y_true_onehot, p_pred, eps=1e-12):
    """Cross-entropy categórica. y_true_onehot: (batch, K)."""
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(np.sum(y_true_onehot * np.log(p_pred),
                                  axis=-1)))


def focal(y_true, p_pred, gamma=2.0, alpha=0.25, eps=1e-12):
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(
        alpha * (1 - p_pred) ** gamma * y_true * np.log(p_pred)
    ))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-loss-elegir
fase: 03
leccion: 05
---

Eres un asistente que ayuda a elegir la función de pérdida.
Recibirás la tarea (regresión/clasificación, presencia de
outliers, desbalance de clases) y la métrica de evaluación.
Tu trabajo:

1. Regresión sin outliers: MSE.
2. Regresión con outliers: MAE o Huber (delta=1).
3. Clasificación binaria: cross-entropy binaria.
4. Clasificación multiclase: cross-entropy categórica.
5. Clasificación con desbalance extremo: focal loss.
6. Embeddings: contrastive o triplet loss.
7. Advertir contra MSE en clasificación (saturación de
   gradiente).
8. Recomendar monitoreo: la pérdida debe bajar
   monotónicamente en training, plateau en val.
```

## Ejercicios

1. **MSE vs MAE**: compara la convergencia de un MLP con
   ambas pérdidas en un dataset con outliers sintéticos.
2. **Focal vs cross-entropy**: aplica a un dataset
   desbalanceado 1:100 y compara F1.
3. **Desafío**: implementa Dice loss para segmentación
   binaria.

## Lecturas recomendadas

- *Deep Learning* — Goodfellow et al. (cap. 8 sobre
  pérdidas).
- *Focal Loss for Dense Object Detection* — Lin et al.,
  2017.
- PyTorch nn.loss: <https://pytorch.org/docs/stable/nn.html#loss-functions>.

---

> 📚 **Adaptación al español** de la lección "[Loss Functions]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
