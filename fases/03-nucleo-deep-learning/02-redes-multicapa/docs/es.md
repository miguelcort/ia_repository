# 02 — Redes multi-capa y forward pass

> Apilar perceptrones en capas resuelve XOR y habilita el aprendizaje de funciones no lineales. La pieza central de toda red neuronal.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-el-perceptron
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar el forward pass de un MLP con `n` capas
  ocultas.
- Componer activaciones y transformaciones lineales.
- Diagnosticar dimensiones de entrada y salida, así como
  funciones de activación por capa.
- Conectar forward pass con la arquitectura moderna
  (Transformer, ResNet, etc.).

## El problema

El perceptrón solo puede resolver problemas linealmente
separables. Para datos reales (imagen, texto, audio) la
frontera de decisión es no lineal. Apilar perceptrones en
capas (un MLP) y entrenarlos con backpropagation resuelve
esto: con una capa oculta de 2 neuronas, XOR se vuelve
separable; con más capas, las redes modernas pueden aprender
cualquier función continua (Universal Approximation
Theorem).

## El concepto

**MLP (Multi-Layer Perceptron).** Una red feedforward de `L`
capas. Cada capa aplica una transformación lineal seguida de
una activación no lineal:

```text
h^0 = x                    # input
h^l = σ^l(W^l h^{l-1} + b^l)  # para l = 1, ..., L
ŷ = h^L                   # output (logits o probabilidades)
```

La no-linealidad de `σ^l` es esencial: sin ella, la
composición de transformaciones lineales es lineal.

**Universal Approximation Theorem (Cybenko, 1989).** Un MLP
con una sola capa oculta y activación sigmoidal puede
aproximar cualquier función continua en un compacto, dado
suficientes neuronas. La profundidad (`L > 1`) permite
representaciones más eficientes.

**Forward pass.** El cómputo de `ŷ = f(x; θ)` donde `θ = {W,
b}` son los parámetros. Es un grafo de cómputo dirigido:
los valores fluyen desde la entrada hasta la salida. Es
necesario tanto para inferencia como para backpropagation
(que calcula gradientes respecto a `θ`).

**Funciones de activación por capa.** No todas las
activaciones son iguales. Reglas prácticas:

- **Capas ocultas:** ReLU es la default. GELU en
  transformers. Tanh en RNNs.
- **Capa de salida:** sigmoid (binaria), softmax
  (multiclase), lineal (regresión), ninguna (embeddings).

**Dimensiones.**

- `x ∈ R^{d_0}`: features de entrada.
- `W^l ∈ R^{d_l × d_{l-1}}`: pesos de la capa `l`.
- `b^l ∈ R^{d_l}`: bias de la capa `l`.
- `h^l ∈ R^{d_l}`: activaciones de la capa `l`.
- Última capa `L`: `d_L` es el número de clases (clasificación)
  o 1 (regresión escalar).

**Inicialización.** Por ahora, aleatoria gaussiana con
varianza pequeña. La lección 08 cubre Xavier y He en
detalle.

**Capacidades representacionales.**

- 1 capa oculta: cualquier función continua.
- 2 capas ocultas: funciones más expresivas, mismo poder
  asintótico.
- Muchas capas (deep): representación jerárquica (cada
  capa aprende features de más alto nivel).

**Cuándo usar MLP.**

| Situación | Recomendación |
|---|---|
| Features tabulares | Sí, primera opción |
| Texto | Mejor transformer |
| Imagen | Mejor CNN o ViT |
| Audio | Mejor CNN 1D o Whisper |
| Series de tiempo | Mejor RNN o transformer temporal |

**Trampas.**

- **Sin normalización:** las activaciones explotan o
  colapsan. BatchNorm, LayerNorm, o estandarización de
  entrada.
- **Activación lineal en capas ocultas:** colapsa a una
  sola capa. Usa ReLU o similar.
- **Demasiadas neuronas con datos pocos:** overfitting
  masivo. Regulariza o usa menos neuronas.

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


ACTIVACIONES = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu}


class MLP:
    """MLP feedforward con L capas. Sin entrenamiento: solo
    forward pass. El entrenamiento está en la lección 03
    (backpropagation)."""

    def __init__(self, dims, activaciones):
        """dims: [d_0, d_1, ..., d_L]. activaciones: lista de L
        strings para capas 1..L (la última puede ser 'none')."""
        self.L = len(dims) - 1
        self.W = [np.random.randn(dims[l], dims[l - 1]) * 0.1
                  for l in range(1, self.L + 1)]
        self.b = [np.zeros(dims[l]) for l in range(1, self.L + 1)]
        self.activaciones = activaciones

    def forward(self, x):
        """x: (batch, d_0). Devuelve (batch, d_L)."""
        h = x
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            act = ACTIVACIONES.get(self.activaciones[l], lambda z: z)
            h = act(z)
        return h
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mlp-arquitectura
fase: 03
leccion: 02
---

Eres un asistente que ayuda a diseñar la arquitectura de un
MLP. Recibirás el tamaño de los datos y el tipo de tarea. Tu
trabajo:

1. Para features tabulares: MLP de 2-3 capas con ReLU.
2. Empezar con [d_0, 128, 64, num_clases] como baseline.
3. Si underfitting: más neuronas o más capas.
4. Si overfitting: dropout, weight decay, o menos
   neuronas.
5. Última capa: sigmoid (binaria), softmax
   (multiclase), lineal (regresión).
6. Inicialización: He para ReLU, Xavier para tanh.
7. Optimizer: AdamW con lr=1e-3 como default.
```

## Ejercicios

1. **MLP con n capas**: implementa un MLP con `L=3` y
   aplícalo a XOR.
2. **Funciones de activación mixtas**: experimenta con
   ReLU en capas ocultas y softmax en la última.
3. **Desafío**: visualiza las activaciones de un MLP
   entrenado en MNIST y observa cómo se especializa cada
   neurona.

## Lecturas recomendadas

- *Deep Learning* — Goodfellow, Bengio, Courville (cap. 6).
- *Neural Networks and Deep Learning* — Michael Nielsen.
- *Dive into Deep Learning* — Zhang, Lipton, Smola.
- PyTorch nn.Module: <https://pytorch.org/docs/stable/nn.html>.

---

> 📚 **Adaptación al español** de la lección "[Multi-layer Networks and Forward Pass]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
