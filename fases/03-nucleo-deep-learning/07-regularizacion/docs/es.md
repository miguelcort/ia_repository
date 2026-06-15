# 07 — Regularización: dropout, weight decay, batch norm

> Sin regularización, las redes modernas sobreajustan casi instantáneamente. Dropout y weight decay son obligatorios; batch norm estabiliza el entrenamiento.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-optimizadores
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar dropout y entender su interpretación como
  ensemble.
- Aplicar L1 y L2 (weight decay) y diagnosticar su efecto.
- Implementar batch normalization y layer normalization.
- Conocer regularización moderna: data augmentation,
  mixup, label smoothing.

## El problema

Tienes una red con millones de parámetros y 100k muestras.
Sin regularización, el modelo sobreajusta: training loss
sigue bajando, val loss empieza a subir. La regularización
limita la capacidad efectiva del modelo. La lección cubre
las tres técnicas canónicas (dropout, weight decay, batch
norm) y las modernas (mixup, label smoothing).

## El concepto

**Weight decay (L2).** Suma `λ ||θ||²` a la pérdida. Penaliza
pesos grandes, lo que induce modelos con pesos pequeños y
distribuidos. En SGD se ve como `θ ← θ - lr · g - lr · λ · θ`.
En Adam, debe ser **decoupled** (AdamW) para que no
interactúe mal con la adaptación del learning rate. Default
en transformers: `λ = 0.01` a `0.1`.

**L1 regularization.** Suma `λ ||θ||₁` en vez de L2. Induce
sparsity (muchos coeficientes exactamente 0). Menos usado en
deep learning que L2, pero útil para interpretabilidad.

**Dropout (Srivastava et al., 2014).** En cada paso de
entrenamiento, "apaga" cada neurona con probabilidad `p`
(típico 0.5 para capas ocultas, 0.1 para input). La red se
entrena con una sub-red aleatoria en cada paso. En
inferencia, todas las neuronas están activas y se
multiplican los pesos por `(1 - p)` para compensar (o
equivalentemente, se divide por `(1 - p)`).

**Interpretación de dropout como ensemble.** Cada paso
entrena una sub-red diferente. Al final, la red completa es
como un ensemble implícito de `2^N` sub-redes. Es una
regularización muy efectiva y casi siempre presente en
redes fully-connected.

**Batch Normalization (Ioffe & Szegedy, 2015).** Normaliza
las activaciones de cada capa sobre el mini-batch:
`x_hat = (x - μ_B) / sqrt(σ²_B + ε)`, luego `y = γ x_hat + β`
(aprendibles). Tiene tres efectos: (1) acelera
convergencia, (2) permite learning rates más altos, (3)
actúa como regularizador. En CNN, típicamente entre conv y
activación. En transformers ha sido reemplazado por
LayerNorm.

**Layer Normalization.** Normaliza sobre las features de
un solo ejemplo (no sobre el batch). Es independiente del
tamaño del batch, lo que es crucial en transformers con
secuencias de longitud variable. Es la default en BERT, GPT,
T5, LLaMA.

**Data augmentation.** Regulariza artificialmente
expandiendo el dataset. Para imagen: rotaciones, flips,
crops, color jitter. Para texto: back-translation,
paráfrasis. Para audio: noise injection, time stretching.
Es la regularización más efectiva en visión.

**Mixup (Zhang et al., 2017).** Entrena con mezclas
lineales de ejemplos: `(x, y) = λ · (x_i, y_i) + (1 - λ) · (x_j,
y_j)`. Fuerza al modelo a ser lineal entre clases, lo que
regulariza fuertemente. Mejora calibración y robustez.

**Label smoothing.** Reemplaza one-hot `[0, 1, 0, ...]` por
`[ε/K, 1 - ε + ε/K, ε/K, ...]` con `ε = 0.1`. Reduce
confianza del modelo, mejora calibración. Usado en muchos
clasificadores de imagen.

**Cuándo usar cada técnica.**

| Técnica | Cuándo |
|---|---|
| Weight decay (L2 / AdamW) | Siempre |
| Dropout | Fully-connected, transformers |
| BatchNorm | CNN |
| LayerNorm | Transformers |
| Data augmentation | Visión, audio |
| Mixup | Clasificación de imagen |
| Label smoothing | Clasificación de imagen, transformers |
| Early stopping | Siempre (es gratis) |

**Trampas.**

- **Dropout + BatchNorm en inferencia:** dropout se
  desactiva, pero BatchNorm usa estadísticas acumuladas, no
  las del batch. Asegúrate de hacer `model.eval()` antes de
  predecir.
- **Weight decay en biases y LayerNorm:** normalmente NO
  aplicas weight decay a `bias` ni a `LayerNorm.weight`. Filtra
  los parámetros antes de aplicar.
- **Demasiada regularización:** subajuste. Monitorea training
  vs val loss; si ambos son altos, reduce.
- **Mixup con tareas de regresión con escala distinta:**
  puede romper. Normaliza primero.

## Constrúyelo

```python
import numpy as np


def dropout(x, p=0.5, training=True):
    """Apaga neuronas con probabilidad p durante training."""
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    return x * mask


def weight_decay_l2(weights, lam=0.01):
    """Suma lambda * ||w||^2 a la pérdida (solo el término)."""
    return lam * sum(np.sum(w ** 2) for w in weights)


def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    """BatchNorm: normaliza sobre el batch (axis 0).
    x: (batch, features)."""
    if training:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean.flatten()
            running_var[:] = (1 - momentum) * running_var + momentum * var.flatten()
        x_hat = (x - mean) / np.sqrt(var + eps)
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    return gamma * x_hat + beta


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm: normaliza sobre las features de un ejemplo."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def mixup(x, y, alpha=0.2):
    """Mezcla lineal de pares de ejemplos."""
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(x))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix, lam
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-regularization
fase: 03
leccion: 07
---

Eres un asistente que ayuda a configurar la regularización.
Recibirás la arquitectura, el tamaño del dataset, y los
síntomas (overfitting, gradientes, etc.). Tu trabajo:

1. Si tienes overfitting: añadir weight decay (0.01-0.1).
2. Si es red fully-connected: añadir dropout (0.1-0.5).
3. Si es transformer: AdamW con weight decay 0.1.
4. Si es CNN: BatchNorm entre conv y activación.
5. Si tienes data augmentation disponible: usar más
   augmentación, no más regularización.
6. Si el dataset es muy pequeño: considerar mixup o
   cutmix.
7. Si loss de val sube mientras train baja: reducir
   capacidad del modelo o aumentar regularización.
8. Advertir contra regularizar en exceso: lleva a
   subajuste.
9. Aplicar weight decay solo a pesos, no a biases ni
   LayerNorm.
```

## Ejercicios

1. **Dropout**: implementa y visualiza la diferencia de
   accuracy con y sin dropout en un dataset pequeño.
2. **BatchNorm vs LayerNorm**: compara la convergencia de
   un MLP con ambos.
3. **Desafío**: implementa mixup en un clasificador de
   imágenes y compara con/sin mixup.

## Lecturas recomendadas

- *Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting* — Srivastava et al., 2014.
- *Batch Normalization* — Ioffe & Szegedy, 2015.
- *mixup: Beyond Empirical Risk Minimization* — Zhang et
  al., 2017.
- PyTorch nn.Dropout: <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>.

---

> 📚 **Adaptación al español** de la lección "[Regularization]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
