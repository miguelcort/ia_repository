# 18 — Selección de features

> Más features no siempre es mejor: malgasta capacidad del modelo, introduce ruido y rompe la interpretabilidad. Saber reducir dimensionalidad es saber.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-feature-engineering-y-seleccion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar varianza threshold.
- Calcular correlación y eliminar features redundantes.
- Aplicar filter methods (chi², ANOVA, mutual information).
- Implementar wrapper methods (forward selection, RFE) y
  embedded methods (Lasso).

## El problema

Tienes 500 features y 1000 muestras. Tu modelo no generaliza
porque con tantas features relativas a las muestras, se
sobreajusta. O tienes 50 features pero 5 son versiones
codificadas de la misma variable, multiplicando su peso
implícito. La selección de features reduce dimensionalidad,
elimina redundancia y mejora interpretabilidad.

## El concepto

**Tres familias de métodos.**

- **Filter:** se aplican antes del modelo, independientemente
  del algoritmo. Rápidos pero ignoran la interacción con el
  modelo. Ejemplos: varianza threshold, correlación, chi²,
  ANOVA, mutual information.
- **Wrapper:** usan un modelo como caja negra para evaluar
  subconjuntos. Más lentos pero capturan la interacción con
  el modelo. Ejemplos: forward selection, backward
  elimination, RFE.
- **Embedded:** la selección ocurre durante el entrenamiento
  del modelo. Ejemplos: Lasso (L1), tree-based importance.

**Varianza threshold.** Elimina features con varianza
cercana a cero: no aportan información. `VarianceThreshold(0.0)`
en sklearn. Útil como primer paso.

**Correlación.** Calcula la matriz de correlación. Si dos
features tienen `|corr| > 0.95`, elimina una (la de menor
correlación con el target). Reduce multicolinealidad.

**Filter methods univariados.**

- **chi² (χ²):** mide dependencia entre feature categórica y
  target categórico. Mayor χ² = más dependencia = más
  relevante.
- **ANOVA F-value:** mide diferencia de medias entre grupos.
  Para features numéricas y target categórico.
- **Mutual information:** mide dependencia general (lineal y
  no lineal). No asume distribución.

**Wrapper methods.**

- **Forward selection:** empieza con 0 features. Añade la
  feature que más mejora la métrica. Repite hasta que no
  haya mejora.
- **Backward elimination:** empieza con todas. Elimina la
  feature cuya remoción menos degrada. Repite.
- **RFE (Recursive Feature Elimination):** ajusta el modelo,
  elimina la feature menos importante, repite. Más estable
  que forward selection.

**Embedded methods.**

- **Lasso (L1):** la regularización L1 lleva algunos
  coeficientes a exactamente 0. Efectivamente selecciona
  features.
- **Tree-based importance:** Random Forest y XGBoost reportan
  feature importance. Útil para una primera criba.

**Cuándo usar cada familia.**

| Familia | Cuándo |
|---|---|
| Filter (varianza, correlación) | Primer paso siempre |
| Filter (chi², ANOVA, MI) | Miles de features, modelo agnóstico |
| Wrapper (RFE) | Decenas de features, modelo fijo |
| Embedded (Lasso) | Cientos de features, modelo lineal |
| Tree-based importance | Cientos de features, modelo arbóreo |

**Trampas.**

- **Selección antes de split:** data leakage. Siempre
  selecciona features solo con train.
- **Threshold arbitrario:** "top 20 features" sin justificación.
  Usa CV para elegir el número.
- **Ignorar multicolinealidad:** mantener features altamente
  correlacionadas duplica su peso implícito.
- **Recalcular selección en cada fold:** genera diferentes
  features por fold. Hazlo una vez en train, aplícalo igual
  en test.

## Constrúyelo

```python
import numpy as np


def varianza_threshold(X, umbral=0.0):
    """Devuelve máscara booleana de features con varianza > umbral."""
    var = X.var(axis=0)
    return var > umbral, var


def correlacion(X, umbral=0.95):
    """Elimina features altamente correlacionadas."""
    corr = np.corrcoef(X.T)
    n = X.shape[1]
    mascara = np.ones(n, dtype=bool)
    for i in range(n):
        if not mascara[i]:
            continue
        for j in range(i + 1, n):
            if mascara[j] and abs(corr[i, j]) > umbral:
                mascara[j] = False
    return mascara


def chi_cuadrado(X, y):
    """chi²: dependencia entre feature categórica y target categórico."""
    from sklearn.feature_selection import chi2
    return chi2(X, y)


def informacion_mutua(X, y):
    """Mutual information (acepta X continua y y discreta)."""
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(X, y, random_state=42)


def forward_selection(model_fn, X, y, n_features=10, cv=5):
    """Forward selection: añade la feature que más mejora el score."""
    from sklearn.model_selection import cross_val_score
    n = X.shape[1]
    selected = []
    remaining = list(range(n))
    best_score = -np.inf
    while len(selected) < n_features and remaining:
        scores = []
        for f in remaining:
            cand = selected + [f]
            modelo = model_fn()
            s = cross_val_score(modelo, X[:, cand], y, cv=cv).mean()
            scores.append((f, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] > best_score:
            best_score = scores[0][1]
            selected.append(scores[0][0])
            remaining.remove(scores[0][0])
        else:
            break  # no mejora
    return selected
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-feature-selection
fase: 02
leccion: 18
---

Eres un asistente que ayuda a diseñar el pipeline de
selección de features. Recibirás el dataset (n, d) y el
modelo. Tu trabajo:

1. Varianza threshold como primer paso (eliminar features
   constantes).
2. Correlación (> 0.95) para eliminar redundantes.
3. Filter (chi², ANOVA, mutual information) si d > 100 y
   el modelo es agnóstico.
4. Lasso (L1) si modelo lineal y d > 50.
5. Tree-based importance (RF, XGBoost) si modelo arbóreo.
6. RFE o forward selection si d < 50 y modelo fijo.
7. Siempre: selección solo con train, aplicar igual en
   test.
8. Validar con CV: la métrica en test debe mejorar vs
   baseline sin selección.
9. Reportar cuántas features se eliminaron y el efecto en
   accuracy.
```

## Ejercicios

1. **Lasso path**: implementa la regularización L1 y observa
   cómo coeficientes caen a cero al variar `α`.
2. **Boruta**: implementa la selección robusta basada en RF
   con shadow features.
3. **Desafío**: RFE con cross-validation sobre un dataset
   realista (MNIST reducido) y compara con selección
   univariada.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 6.
- *Feature Selection for Data Mining* — Liu & Motoda.
- scikit-learn feature_selection: <https://scikit-learn.org/stable/modules/feature_selection.html>.
- *Feature Engineering and Selection* — Kuhn & Johnson.

---

> 📚 **Adaptación al español** de la lección "[Feature Selection]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
