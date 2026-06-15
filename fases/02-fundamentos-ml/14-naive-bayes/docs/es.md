# 14 — Naive Bayes

> El clasificador más viejo de ML sigue siendo relevante: rápido, robusto, base para spam, sentimiento y categorización de texto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar GaussianNB (features continuas, asumiendo
  normalidad).
- Implementar MultinomialNB (conteos, con Laplace smoothing).
- Diagnosticar cuándo Naive Bayes es la opción correcta.
- Aplicar ComplementNB para datos desbalanceados.

## El problema

Necesitas clasificar 100k documentos en 5 categorías en menos
de 1 minuto de CPU. Una red neuronal entrenaría por horas;
XGBoost requiere feature engineering. Naive Bayes con
TF-IDF entrena en milisegundos y a menudo supera a modelos
más complejos en clasificación de texto. La "ingenuidad" de
asumir independencia condicional entre features es lo que lo
hace rápido y sorprendentemente bueno.

## El concepto

**Regla de Bayes.** Para features `x = (x_1, ..., x_d)` y
clase `c`:

```text
P(c | x) = P(x | c) · P(c) / P(x)
```

La predicción es `argmax_c P(c | x)`. Como `P(x)` no depende
de `c`, basta con maximizar `P(x | c) · P(c)`.

**Suposición naive.** Asumir independencia condicional entre
features dado la clase:

```text
P(x | c) = Π_i P(x_i | c)
```

Es lo que da el nombre. Casi nunca se cumple en la práctica,
pero igual funciona bien.

**Tipos de Naive Bayes según el tipo de feature.**

- **GaussianNB:** features continuas, asume `P(x_i | c) ~ N(μ_{c,i}, σ²_{c,i})`.
  Estima `μ` y `σ` por clase y feature.
- **MultinomialNB:** features de conteo (bolsa de palabras,
  n-gramas). `P(x_i | c) ∝ count(x_i, c)`. Usa Laplace
  smoothing para evitar probabilidades cero.
- **BernoulliNB:** features binarias (presencia/ausencia de
  palabra). `P(x_i | c)` con dos valores (presente, ausente).
- **ComplementNB:** variante de MultinomialNB para datos
  desbalanceados. Usa estadísticas de la complement class
  (todas las clases excepto c).

**Laplace smoothing.** Evita el problema de probabilidad cero
cuando un feature nunca aparece con una clase. Suma 1 a cada
conteo (o α en general):

```text
P(x_i = v | c) = (count(x_i = v, c) + α) / (Σ_v count(x_i = v, c) + α · |V|)
```

Sin smoothing, una sola palabra ausente en el training set
puede hacer que la predicción sea cero para toda una clase.

**Cuándo usar Naive Bayes.**

| Situación | Naive Bayes |
|---|---|
| Texto con bolsa de palabras o TF-IDF | Sí, baseline fuerte |
| Features categóricas independientes | Sí |
| Dataset muy grande (millones) | Sí, escala lineal |
| Necesidad de probabilidades calibradas | Sí, output es probabilístico |
| Features altamente correlacionadas | No, mejor regresión logística |
| Datos de imagen | No, mejor CNN |

**Trampas.**

- **Asumir GaussianNB para datos no gaussianos:** ajusta
  mal. Usa la distribución correcta.
- **Ignorar Laplace smoothing:** sin él, una palabra
  desconocida mata la predicción.
- **MultinomialNB con TF-IDF normalizado por L1:** no es
  el uso correcto. MultinomialNB espera conteos crudos o
  TF sin normalizar.

**Naive Bayes vs regresión logística.** Cuando se cumplen las
suposiciones de NB, NB gana (más rápido, menos datos). Cuando
no se cumplen, regresión logística es mejor. En la práctica,
en clasificación de texto con suficientes datos, la
diferencia es marginal.

## Constrúyelo

```python
import numpy as np


class GaussianNB:
    def fit(self, X, y):
        self.clases = np.unique(y)
        self.mu = []
        self.sigma = []
        self.prior = []
        for c in self.clases:
            Xc = X[y == c]
            self.mu.append(Xc.mean(axis=0))
            self.sigma.append(Xc.std(axis=0) + 1e-9)
            self.prior.append(len(Xc) / len(X))
        return self

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        log_probs = []
        for c, mu, sigma, prior in zip(
            self.clases, self.mu, self.sigma, self.prior
        ):
            log_p = np.log(prior)
            log_p += -0.5 * np.log(2 * np.pi * sigma ** 2)
            log_p += -0.5 * ((x - mu) / sigma) ** 2
            log_probs.append(log_p.sum())
        return self.clases[np.argmax(log_probs)]


class MultinomialNB:
    def fit(self, X, y, alpha=1.0):
        self.clases = np.unique(y)
        self.alpha = alpha
        self.log_prob = []
        for c in self.clases:
            Xc = X[y == c]
            count = Xc.sum(axis=0) + alpha
            self.log_prob.append(np.log(count / count.sum()))
        return self

    def predict(self, X):
        return np.array([
            self.clases[np.argmax(X[i] @ np.array(self.log_prob).T)]
            for i in range(X.shape[0])
        ])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nb-elegir
fase: 02
leccion: 14
---

Eres un asistente que ayuda a elegir el clasificador Naive
Bayes apropiado. Recibirás la descripción de los features
(tipo, distribución, cardinalidad) y el balance de clases.
Tu trabajo:

1. Features continuas, asumibles gaussianas: GaussianNB.
2. Conteos (palabras, frecuencias): MultinomialNB.
3. Features binarias (presencia/ausencia): BernoulliNB.
4. Clases desbalanceadas: ComplementNB o `class_prior`.
5. Si las features están altamente correlacionadas: advertir
   que Naive Bayes no es la mejor opción.
6. Sugerir Laplace smoothing (alpha=1) para evitar
   probabilidades cero.
7. Reportar log-loss y accuracy sobre validación.
```

## Ejercicios

1. **BernoulliNB**: implementa la versión para features
   binarias.
2. **TF-IDF + MultinomialNB**: clasificador de spam completo
   sobre un dataset como SMS Spam Collection.
3. **Desafío**: compara NB vs LogReg en el dataset
   20newsgroups y reporta accuracy, F1, y tiempo de
   entrenamiento.

## Lecturas recomendadas

- *Speech and Language Processing* — Jurafsky & Martin, cap. 4.
- *Pattern Recognition and Machine Learning* — Bishop.
- scikit-learn naive_bayes: <https://scikit-learn.org/stable/modules/naive_bayes.html>.
- *Introduction to Information Retrieval* — Manning,
  Raghavan, Schütze.

---

> 📚 **Adaptación al español** de la lección "[Naive Bayes]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
