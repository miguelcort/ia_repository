# Fase 2 — Fundamentos de Machine Learning

> ML clásico: la columna vertebral de la mayor parte de la IA en producción.

Esta fase cubre el **ML clásico supervisado y no supervisado** que
sigue explicando la mayor parte de los sistemas en producción:
regresión lineal para forecasting de demanda, regresión logística
para scoring de fraude, Random Forest para scoring crediticio,
K-Means para segmentación de clientes, series de tiempo para
predicción de inventario. Aunque las redes profundas dominan la
discusión pública, los algoritmos de esta fase siguen siendo la
elección rentable cuando los datos son tabulares, los requisitos
de interpretabilidad son altos, o el costo de entrenar una red
neuronal no se justifica.

La pedagogía es **construir antes de usar**: cada algoritmo se
implementa primero en NumPy (o en Python puro cuando la idea es
pedagógica) y luego se compara con la versión de `scikit-learn`. La
implementación propia es breve, finita y verificable; el contraste
con la librería muestra dónde el framework añade valor (regularización
incorporada, optimizadores eficientes, paralelismo) y dónde tu versión
es suficiente (datasets pequeños, prototipos, enseñanza).

## Índice de lecciones

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [¿Qué es Machine Learning?](01-que-es-machine-learning/) | Aprender | Definición, taxonomía y cuándo NO usar ML. |
| 02 | [Regresión lineal desde cero](02-regresion-lineal-desde-cero/) | Construir | GD batch, normal equation y diagnóstico de residuales. |
| 03 | [Regresión logística y clasificación](03-regresion-logistica/) | Construir | Función sigmoide, entropía cruzada binaria y multiclase. |
| 04 | [Árboles de decisión y Random Forest](04-arboles-de-decision-random-forest/) | Construir | Gini, entropía, *bagging* y feature importance. |
| 05 | [Support Vector Machines (SVM)](05-support-vector-machines/) | Construir | Margen máximo, *kernel trick* y SVM lineal desde cero. |
| 06 | [KNN y métricas de distancia](06-knn-y-distancias/) | Construir | k-vecinos con distintas métricas y validación de k. |
| 07 | [Aprendizaje no supervisado: K-Means, DBSCAN](07-aprendizaje-no-supervisado/) | Construir | Clustering con K-Means++ y DBSCAN para ruido. |
| 08 | [Feature engineering y selección](08-feature-engineering-y-seleccion/) | Construir | Encoding, escalado, *binning* e información mutua. |
| 09 | [Métricas y validación cruzada](09-metricas-y-validacion-cruzada/) | Construir | Accuracy, F1, AUC, k-fold, stratified, time-series split. |
| 10 | [Sesgo, varianza y curva de aprendizaje](10-sesgo-varianza-y-curva-de-aprendizaje/) | Aprender | Diagnóstico de over/underfitting con `learning_curve`. |
| 11 | [Métodos de ensemble: boosting, bagging, stacking](11-ensemble-methods/) | Construir | AdaBoost, Gradient Boosting y Voting Classifier. |
| 12 | [Ajuste de hiperparámetros](12-ajuste-de-hiperparametros/) | Construir | Grid, Random, Bayesian search con Optuna. |
| 13 | [Pipelines de ML y tracking de experimentos](13-pipelines-de-ml-y-tracking/) | Construir | `sklearn.Pipeline`, MLflow, versionado de datasets. |
| 14 | [Naive Bayes](14-naive-bayes/) | Construir | Bayes ingenuo multinomial y aplicación a texto. |
| 15 | [Series de tiempo](15-series-de-tiempo/) | Construir | ARIMA, Prophet, descomposición STL. |
| 16 | [Detección de anomalías](16-deteccion-de-anomalias/) | Construir | Isolation Forest, LOF y umbrales estadísticos. |
| 17 | [Datos desbalanceados](17-datos-desbalanceados/) | Construir | SMOTE, undersampling, *class weights* y threshold tuning. |
| 18 | [Selección de features](18-seleccion-de-features/) | Construir | Filter, wrapper y embedded con `sklearn.feature_selection`. |

## Prerrequisitos

- **Fase 0** completa (entorno, Git, Python, Docker).
- **Fase 1** completa (álgebra lineal, cálculo, probabilidad) —
  *muy recomendada*. La Fase 2 funciona sin ella, pero las
  explicaciones serán más densas.
- Conocimiento básico de pandas (cargar CSV, groupby, merge).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Construir** desde cero un modelo de regresión lineal con GD
  y compararlo con la solución analítica.
- **Clasificar** con regresión logística, SVM, KNN, árboles y
  Random Forest, eligiendo el más adecuado según el problema.
- **Agrupar** datos con K-Means, DBSCAN y clustering jerárquico,
  evaluando con silueta y Davies-Bouldin.
- **Diagnosticar** sobreajuste y subajuste con curvas de
  aprendizaje y seleccionar el modelo con la métrica correcta.
- **Optimizar** hiperparámetros con grid, random y búsqueda
  bayesiana, registrando cada experimento en MLflow.
- **Manejar** clases desbalanceadas con SMOTE, pesos y
  ajuste de umbral.
- **Explicar** predicciones con feature importance, permutation
  importance y SHAP (introductorio, la Fase 18 profundiza).

## Stack y herramientas

- **NumPy, pandas, scikit-learn** como núcleo de la fase.
- **Matplotlib, seaborn** para diagnóstico visual.
- **XGBoost, LightGBM** en la lección de ensembles y boosting.
- **MLflow** para tracking de experimentos.
- **Optuna** para búsqueda bayesiana de hiperparámetros.
- **Prophet** (opcional) para series de tiempo.
- **imbalanced-learn** para SMOTE y undersampling.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Regresión lineal** | Lecciones 2, 8, 13 | Fase 3 (regresión neuronal), Fase 11 (LoRA) |
| **Función de pérdida** | Lecciones 2, 3, 5, 14 | Fase 3 (MSE, cross-entropy) |
| **Overfitting / underfitting** | Lección 10 | Todas las fases siguientes |
| **Validación cruzada** | Lección 9 | Fase 3, Fase 10, Fase 19 |
| **Feature engineering** | Lecciones 8, 18 | Fase 4 (Visión), Fase 5 (NLP) |
| **Ensemble** | Lección 11 | Fase 3 (dropout como ensemble implícito) |
| **Series de tiempo** | Lección 15 | Fase 17 (drift detection) |
| **Anomalías** | Lección 16 | Fase 17 (monitoreo) |

## Cómo estudiar esta fase

1. **Empieza por la lección 02** (regresión lineal). Aunque la
   lección 01 da contexto, es la 02 la que te pone a programar la
   primera actualización de pesos.
2. **Resuelve cada demo a mano en una libreta antes de correr el
   código.** Esto fuerza la intuición sobre la forma de las
   operaciones matriciales.
3. **Compara siempre tu implementación con scikit-learn.** La
   lección 02 termina con una tabla: `tu RMSE` vs `LinearRegression
   RMSE`. Si difieren en más de 1e-6, vuelve al código.
4. **No saltes la lección 10** (sesgo-varianza). Es la base
   conceptual de toda decisión arquitectónica en las fases 3+.
5. **MLflow desde la lección 13.** Una vez que lo configures,
   úsalo en todas las prácticas siguientes; el *costo de
   adopción* es bajo y el beneficio, permanente.

## Verificación de progreso

```bash
# Lección 02 — regresión lineal reproducible
python3 fases/02-fundamentos-ml/02-regresion-lineal-desde-cero/code/main.py

# Lección 09 — validación cruzada estratificada
python3 fases/02-fundamentos-ml/09-metricas-y-validacion-cruzada/code/main.py

# Lección 13 — pipeline + MLflow
python3 fases/02-fundamentos-ml/13-pipelines-de-ml-y-tracking/code/main.py
```

Si los tres demos terminan con código 0 y el RMSE de la lección 02
coincide con scikit-learn en al menos 4 decimales, el estudiante
está listo para la Fase 3.

## Cuándo NO usar ML clásico

| Situación | Modelo clásico | Alternativa |
|---|---|---|
| Datos tabulares pequeños/medianos | Regresión, Random Forest, XGBoost | — (quedarse aquí) |
| Texto en lenguaje natural | Naive Bayes, regresión logística | Fase 5 (transformers) |
| Imágenes | — | Fase 4 (CNN, ViT) |
| Audio crudo | — | Fase 6 (CNN 1D, Whisper) |
| Series de tiempo con patrones complejos | ARIMA, Prophet | RNN, Transformer (Fase 7+) |
| Decisiones secuenciales | — | Fase 9 (RL) |
| Datos multimodales | — | Fase 12 (CLIP, VLMs) |

## Conexión con otras fases

- **Entrada** → [Fase 1 — Fundamentos de matemáticas](../01-fundamentos-matematicas/README.md).
- **Salida natural** → [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md),
  donde cada algoritmo de esta fase se reescribe como red neuronal.
- **Reuso** → Fase 17 (monitoreo reusa detección de anomalías),
  Fase 18 (sesgos reusa muestreo y datos desbalanceados),
  Fase 19 (capstone integra todo).

## Recursos recomendados

- *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani (PDF libre).
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman.
- *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* — Géron.
- *Machine Learning with R* — cuando se necesite base teórica más profunda.
- Documentación oficial: <https://scikit-learn.org>.

## Véase también

- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.
- [glosario/terminos.md](../../glosario/terminos.md) — definiciones canónicas.
- [PLANTILLA_LECCION.md](../../PLANTILLA_LECCION.md) — estructura de cada lección.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
