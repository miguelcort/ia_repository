# 13 — Pipelines de ML y tracking de experimentos

> Sin reproducibilidad, un experimento de ML es un cuento. Pipelines y tracking son el seguro de vida.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-feature-engineering-y-seleccion,
                  09-metricas-y-validacion-cruzada,
                  12-ajuste-de-hiperparametros
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Construir `sklearn.Pipeline` reproducible end-to-end.
- Usar `ColumnTransformer` para aplicar transforms distintos a
  columnas distintas.
- Trackear experimentos con MLflow (parámetros, métricas,
  artefactos).
- Versionar datasets con DVC para reproducibilidad total.

## El problema

Tu modelo entrenó con 0.94 de accuracy. Tres meses después
quieres reproducirlo. El codigo se perdió, las versiones de
las bibliotecas cambiaron, los datos crudos no están, y el
*notebook* tiene celdas ejecutadas en orden distinto. Los
pipelines y el tracking de experimentos son la solución: todo
lo necesario para reproducir está capturado automáticamente
— versión del codigo, de los datos, hiperparámetros,
métricas, y artefactos.

## El concepto

**`sklearn.Pipeline`.** Encadena transforms y un estimador
final en un solo objeto. Se fitea con `fit(X, y)` y predice
con `predict(X)`. Todos los transforms intermedios se fitean
solo con train y se aplican a train y test, eliminando
manualmente la fuente #1 de data leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

**`ColumnTransformer`.** Aplica transforms distintos a
columnas distintas (numéricas vs categóricas) y concatena los
resultados. Esencial cuando tu dataset tiene tipos mezclados.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), categorical_cols),
])
```

**`GridSearchCV` con pipeline.** El pipeline completo se
pasa al grid search, y los hiperparámetros se referencian con
`step__param`. Así puedes tunear el preprocesamiento y el
modelo juntos.

**Tracking con MLflow.** MLflow es el estándar open source
para tracking de experimentos. Cada run registra:

- **Parámetros:** hiperparámetros del modelo.
- **Métricas:** accuracy, F1, AUC, etc.
- **Artefactos:** modelos, gráficos, datasets, archivos
  arbitrarios.
- **Metadatos:** git commit, environment, tags.

El MLflow Tracking Server centraliza los runs; el Model
Registry versiona y stagea modelos (Staging, Production,
Archived). MLflow UI permite comparar runs visualmente.

**Versionado de datos con DVC.** Git no maneja bien datasets
grandes. DVC (Data Version Control) almacena los datos
remotamente (S3, GCS) y mantiene un puntero en Git. Así puedes
recrear un experimento usando exactamente el dataset que se
usó originalmente, incluso si el dataset cambió después.

**Weights & Biases (W&B).** Alternativa a MLflow con mejor
UI y herramientas de colaboración. Popular en deep learning.

**Patrón de un pipeline de ML en producción.**

1. **Ingestión:** cargar datos crudos de la fuente
   (DataWarehouse, Kafka, S3).
2. **Validación:** Great Expectations, TFX, Pandera.
3. **Preprocesamiento:** Pipeline + ColumnTransformer.
4. **Entrenamiento:** tuneo con Optuna + tracking con
   MLflow.
5. **Evaluación:** métricas en test set, fairness checks.
6. **Registro:** subir el modelo al Model Registry.
7. **Despliegue:** endpoint REST, batch, o streaming.
8. **Monitoreo:** drift detection, performance tracking.

**Trampas.**

- **Pipeline no determinista:** siempre fija `random_state`
  en cada step.
- **Preprocesamiento fuera del pipeline:** "solo aplico
  StandardScaler una vez" es la receta del leakage.
- **Sin tracking:** "no sé qué cambió" es la frase que
  precede a días de debugging.
- **Sin versionado de datos:** el dataset cambió, tu
  modelo ya no aplica.

## Constrúyelo

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score


def construir_pipeline(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def tracking_simulado(params, metrics, run_name="run"):
    """Mock de MLflow para entender el patrón sin servidor."""
    run = {"name": run_name, "params": params, "metrics": metrics}
    print(f"[mlflow] {run_name}: {params} -> {metrics}")
    return run
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ml-pipeline
fase: 02
leccion: 13
---

Eres un asistente que ayuda a diseñar pipelines de ML reprodu-
cibles. Recibirás la descripción del dataset y del modelo. Tu
trabajo:

1. Definir preprocesamiento con `ColumnTransformer` y
   separar numéricas vs categóricas.
2. Encadenar todo en `Pipeline` para evitar leakage.
3. Trackear cada run con MLflow: parámetros, métricas,
   artefactos, git commit.
4. Versionar datasets con DVC.
5. Si el modelo va a producción: registrar en Model
   Registry con stage (Staging → Production).
6. Logging de artefactos: archivo del modelo, gráfico de
   métricas, sample de predicciones.
7. Documentar el pipeline en un README.
```

## Ejercicios

1. **Pipeline completo**: construye un `Pipeline` con
   preprocesamiento, features derivados polinomiales y un
   modelo. Usa `GridSearchCV` sobre hiperparámetros del
   pipeline.
2. **MLflow local**: corre un experimento con MLflow
   tracking local y compara runs en la UI.
3. **Desafío**: versiona un dataset con DVC y demuestra que
   puedes cambiar entre versiones.

## Lecturas recomendadas

- *Designing Machine Learning Systems* — Huyen (caps. sobre
  pipelines y tracking).
- scikit-learn Pipeline: <https://scikit-learn.org/stable/modules/compose.html>.
- MLflow: <https://mlflow.org>.
- DVC: <https://dvc.org>.
- Weights & Biases: <https://wandb.ai>.

---

> 📚 **Adaptación al español** de la lección "[ML Pipelines and Experiment Tracking]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
