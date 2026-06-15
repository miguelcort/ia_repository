# 08 — Feature engineering y selección

> Features importan más que el modelo. La misma accuracy se puede lograr con un modelo mediocre y buenas features, o con un modelo excelente y features malas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Aplicar técnicas de encoding para variables categóricas
  (one-hot, target, ordinal).
- Escalar y transformar features (StandardScaler,
  MinMaxScaler, log, Box-Cox).
- Calcular features derivados (interacciones, polinomios,
  fechas, agregaciones).
- Diagnosticar el *pipeline* de features correcto.

## El problema

Tienes una columna `fecha_de_nacimiento` y necesitas la
*edad* como feature. Tienes una columna `categoría` con 50
valores únicos y no sabes cómo meterla al modelo. Tienes
features en escalas muy distintas (ingresos en miles, edad
en años) y el modelo no converge. Feature engineering
resuelve todo eso: convierte datos crudos en features que
los modelos pueden aprovechar.

## El concepto

**Encoding de variables categóricas.**

- **One-hot encoding:** crea una columna binaria por cada
  categoría. `pd.get_dummies()` o `sklearn.preprocessing.OneHotEncoder`.
  Problema: explota con alta cardinalidad (columnas con
  miles de valores únicos).
- **Target encoding:** reemplaza la categoría con la media
  de la variable objetivo. `mean(y | x = categoría)`. Útil
  para alta cardinalidad pero riesgo de *data leakage* si no
  se hace con cross-validation.
- **Ordinal encoding:** asigna un entero a cada categoría.
  Solo si las categorías tienen un orden natural (bajo <
  medio < alto).
- **Frequency encoding:** usa la frecuencia de la categoría.
  Cuando la frecuencia correlaciona con el objetivo.

**Escalado de features numéricos.**

- **StandardScaler (z-score):** `(x - mean) / std`. Media 0,
  varianza 1. Asume distribución gaussiana.
- **MinMaxScaler:** `(x - min) / (max - min)`. Rango [0, 1].
  Sensible a outliers.
- **RobustScaler:** `(x - median) / IQR`. Robusto a outliers.
- **Log/Box-Cox:** transformaciones para datos con colas
  largas (ingresos, frecuencias). Reduce el impacto de
  outliers.

**¿Cuándo escalar?**

| Algoritmo | ¿Escalar? |
|---|---|
| Regresión logística, SVM, KNN, redes neuronales | Sí |
| Árboles, Random Forest, XGBoost | No (son invariantes a escala) |
| PCA, t-SNE, UMAP | Sí |

**Features derivados.**

- **Interacciones:** `x1 * x2`, `x1 / x2`. Capturan relaciones
  no lineales para modelos lineales.
- **Polinomios:** `x^2`, `x^3`. `sklearn.preprocessing.PolynomialFeatures`.
- **Features de fecha:** día de la semana, mes, hora,
  festividad, minutos desde último evento.
- **Features de agregación:** media móvil, conteo de eventos
  en ventana, ranking.
- **Features de texto:** longitud, número de mayúsculas,
  proporción de dígitos, embeddings.

**Pipeline de features.** El orden importa:

1. Manejar valores faltantes (imputar o flag).
2. Encoding categórico.
3. Generación de features derivados.
4. Escalado.
5. Selección de features (opcional, en fase 18).

`scikit-learn` ofrece `Pipeline` y `ColumnTransformer` para
encadenar todo de forma reproducible.

**Trampas comunes.**

- *Data leakage*: calcular `mean` o `std` con todo el dataset
  antes de split. Siempre fit el scaler con train, transform
  train y test por separado.
- *One-hot en alta cardinalidad*: 10k categorías = 10k
  columnas. Usa target o frequency encoding.
- *Escalado después de train/test split*: el scaler debe ver
  solo train. Usar `Pipeline` lo garantiza.

## Constrúyelo

```python
import numpy as np
import pandas as pd


def one_hot(df, columna):
    """One-hot encoding usando pandas."""
    return pd.get_dummies(df[columna], prefix=columna)


def target_encoding(y_por_categoria, suavizado=10):
    """Target encoding con suavizado bayesiano.
    y_por_categoria: dict categoria -> [y1, y2, ...]"""
    media_global = np.mean([y for ys in y_por_categoria.values() for y in ys])
    encoded = {}
    for cat, ys in y_por_categoria.items():
        n = len(ys)
        media = np.mean(ys)
        # Suavizado: (n * media + suavizado * media_global) / (n + suavizado)
        encoded[cat] = (n * media + suavizado * media_global) / (n + suavizado)
    return encoded


def interaccion(*features):
    """Producto de dos o más features."""
    result = features[0]
    for f in features[1:]:
        result = result * f
    return result


def features_fecha(serie):
    """Extrae mes, día de semana, hora desde datetime."""
    return pd.DataFrame({
        "mes": serie.dt.month,
        "dia_semana": serie.dt.dayofweek,
        "hora": serie.dt.hour,
        "es_fin_de_semana": serie.dt.dayofweek.isin([5, 6]).astype(int),
    })


def winsorize(x, limite=0.01):
    """Clip outliers a los cuantiles limite y 1-limite."""
    bajo = np.quantile(x, limite)
    alto = np.quantile(x, 1 - limite)
    return np.clip(x, bajo, alto)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-feature-engineering
fase: 02
leccion: 08
---

Eres un asistente que ayuda a diseñar el pipeline de feature
engineering. Recibirás la descripción del dataset (columnas,
tipos, cardinalidad, valores faltantes). Tu trabajo:

1. Categóricas con cardinalidad < 20: one-hot encoding.
2. Categóricas con cardinalidad alta: target encoding con
   k-fold para evitar leakage.
3. Numéricas con outliers: RobustScaler o winsorize.
4. Numéricas con distribución sesgada: log o Box-Cox.
5. Fechas: extraer mes, día, hora, festividad, recencia.
6. Interacciones para modelos lineales: probar productos
   entre las features más importantes.
7. Validar con cross-validation: nunca fit con todo el
   dataset.
8. Documentar el pipeline en `sklearn.Pipeline` para
   reproducibilidad.
```

## Ejercicios

1. **Pipeline completo**: implementa un `sklearn.Pipeline`
   con imputer, encoder, scaler y modelo.
2. **Target encoding con k-fold**: implementa target
   encoding que use solo folds de train para evitar leakage.
3. **Desafío**: implementa feature engineering para el
   dataset Titanic (o similar) y compara accuracy con/sin
   features derivados.

## Lecturas recomendadas

- *Feature Engineering for Machine Learning* — Zheng & Casari.
- *An Introduction to Statistical Learning* — cap. 5.
- scikit-learn preprocessing: <https://scikit-learn.org/stable/modules/preprocessing.html>.
- *Feature Engineering and Selection* — Kuhn & Johnson.

---

> 📚 **Adaptación al español** de la lección "[Feature Engineering and Selection]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
