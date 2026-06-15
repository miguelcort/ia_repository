# 01 — El perceptrón

> El primer algoritmo que aprendió de datos (Rosenblatt, 1958). Limitado a separables lineales, pero el bloque constructor de todo lo que vino después.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero,
                  03-regresion-logistica
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar el perceptrón de Rosenblatt con la regla de
  aprendizaje Hebbiano.
- Verificar que aprende AND y OR pero no XOR.
- Diagnosticar si un dataset es linealmente separable.
- Conectar el perceptrón con la neurona moderna y
  motivación para el perceptrón multi-capa.

## El problema

Antes de 1958 no existía un algoritmo que aprendiera
*directamente* de los datos para hacer clasificación
binaria. La regresión logística y LDA existían pero con
suposiciones distribucionales. Frank Rosenblatt, inspirado
en la neurona de McCulloch-Pitts, propuso un modelo
biológicamente inspirado: una neurona artificial que ajusta
sus pesos según los errores. La limitación que mostró
Minsky y Papert en 1969 (no aprende XOR) llevó al
"invierno de la IA" y solo se resolvió con el
backpropagation de Rumelhart, Hinton y Williams en 1986.

## El concepto

**Neurona de McCulloch-Pitts (1943).** Una neurona binaria
que recibe entradas `x_1, ..., x_d`, calcula la suma
ponderada `z = Σ w_i x_i + b`, y emite `1` si `z >= 0`, `0`
en caso contrario. Es el **modelo lineal binario** por
excelencia.

**Regla del perceptrón (Rosenblatt, 1958).** Dado un
ejemplo `(x, y)` con `y ∈ {-1, +1}`:

```text
ŷ = sign(w·x + b)
si ŷ != y:
    w ← w + lr · y · x
    b ← b + lr · y
```

La regla es Hebbiana: "los pesos crecen cuando la entrada
y la salida correcta están activas". Convergencia
garantizada si los datos son **linealmente separables** (y
con una cota: el número de updates es `O(R²/γ²)` donde `R`
es la norma máxima de los ejemplos y `γ` es el margen).

**Limitación: XOR.** Minsky y Papert (1969) demostraron que
el perceptrón **no puede aprender XOR**, porque XOR no es
linealmente separable. La salida es `(0, 1, 1, 0)` para
entradas `(0,0), (0,1), (1,0), (1,1)`. No existe una línea
que separe los dos `0`s de los dos `1`s. Esto desencadenó
el primer invierno de la IA.

**Solución: perceptrón multi-capa (MLP).** Apilar perceptrones
en capas y entrenarlos con backpropagation. Con una capa
oculta de 2 neuronas, XOR se vuelve separable. Este es el
origen de las redes neuronales modernas.

**Conexión con la neurona moderna.** La neurona de un MLP
moderno es esencialmente la misma:

```text
z = W x + b
h = σ(z)  # activación no lineal
```

El perceptrón usa `step` (no diferenciable); las redes
modernas usan `sigmoid`, `tanh` o `ReLU` (diferenciables).
Eso es lo que hace posible el backprop.

**Variantes del perceptrón.**

- **Pocket algorithm:** guarda la mejor configuración de
  pesos vista durante el entrenamiento (útil cuando los
  datos no son perfectamente separables).
- **Voted perceptron:** promedia todos los pesos vistos
  durante el entrenamiento, ponderados por su supervivencia.
  Da un clasificador más estable.
- **Perceptrón con margen:** `update` solo si la confianza
  (margen) es menor que cierto threshold.

**Cuándo usar perceptrón hoy.**

| Situación | Recomendación |
|---|---|
| Aprender historia de ML | Sí, lectura |
| Dataset separable lineal | Regresión logística es mejor calibrada |
| Dataset no separable | MLP con backprop |
| Texto clásico | SVM o regresión logística con SGD |

**Trampas.**

- **Esperar convergencia en datos no separables:** oscila
  para siempre. Usa pocket o voted perceptron.
- **No normalizar features:** los pesos se vuelven
  arbitrarios. StandardScaler antes de fit.
- **Usar `step` activation en backprop:** no diferenciable.
  Usa sigmoid o ReLU.

## Constrúyelo

```python
import numpy as np


class Perceptron:
    """Perceptrón de Rosenblatt con etiquetas {-1, +1}.
    Convergencia garantizada solo si los datos son
    linealmente separables."""

    def __init__(self, lr=1.0, n_epocas=100, semilla=0):
        self.lr = lr
        self.n_epocas = n_epocas
        self.semilla = semilla

    def fit(self, X, y):
        # Convertir etiquetas a {-1, +1} si vienen en {0, 1}
        y_p = np.where(y == 0, -1, 1).astype(float)
        rng = np.random.default_rng(self.semilla)
        n, d = X.shape
        self.pesos = rng.normal(0, 0.01, d)
        self.bias = 0.0
        for _ in range(self.n_epocas):
            errores = 0
            for xi, yi in zip(X, y_p):
                margen = yi * (xi @ self.pesos + self.bias)
                if margen <= 0:
                    self.pesos += self.lr * yi * xi
                    self.bias += self.lr * yi
                    errores += 1
            if errores == 0:
                break
        return self

    def predecir(self, X):
        return np.sign(X @ self.pesos + self.bias)

    def predecir_etiquetas(self, X):
        """Devuelve etiquetas en {0, 1}."""
        return (self.predecir(X) > 0).astype(int)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-perceptron
fase: 03
leccion: 01
---

Eres un asistente que ayuda a diagnosticar cuándo usar el
perceptrón clásico vs. alternativas modernas. Recibirás el
dataset y la accuracy. Tu trabajo:

1. Verificar si los datos son linealmente separables
   (visualización 2D, o resolver vía LP).
2. Si no son separables: usar MLP, SVM con kernel, o
   regresión logística.
3. Practica: regresión logística es preferible al
   perceptrón clásico porque da probabilidades calibradas.
4. Para conjuntos grandes (> 10k): perceptrón con SGD
   (sklearn.linear_model.Perceptron) es rápido.
5. Advertir contra esperar convergencia en datos no
   separables: oscila para siempre.
6. Monitorear el número de errores por época.
```

## Ejercicios

1. **Pocket algorithm**: implementar la variante que guarda
   los mejores pesos vistos durante el entrenamiento.
2. **Voted perceptron**: promediar todos los pesos vistos,
   ponderados por su supervivencia.
3. **Desafío**: visualizar la frontera de decisión en 2D
   para AND, OR, NAND, XOR.

## Lecturas recomendadas

- *The Perceptron* — Rosenblatt, 1958 (paper fundacional).
- *Perceptrons* — Minsky & Papert, 1969 (análisis de
  limitaciones).
- *Learning representations by back-propagating errors* —
  Rumelhart, Hinton, Williams, 1986.
- *Neural Networks and Deep Learning* — Michael Nielsen
  (libre en línea).

---

> 📚 **Adaptación al español** de la lección "[The Perceptron]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
