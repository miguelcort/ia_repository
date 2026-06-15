# 16 — Detección de anomalías

> Identificar el 0.1% de eventos que no encajan en el patrón. Fraude, fallas de sensores, intrusiones, valores atípicos en datos de salud.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-metricas-y-validacion-cruzada
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar detección basada en distribución (z-score, IQR).
- Aplicar Isolation Forest y Local Outlier Factor.
- Diagnosticar la diferencia entre outliers y anomalías
  contextuales.
- Evaluar con métricas de anomalías: precision@k y recall.

## El problema

Tienes 10M de transacciones de tarjeta de crédito. El 99.9%
son legítimas y el 0.1% son fraudulentas. No tienes etiquetas
para entrenar. Incluso si las tuvieras, el desbalance es tan
extremo que un clasificador estándar falla. Los algoritmos de
detección de anomalías identifican el 0.1% de eventos que no
encajan en el patrón, sin necesidad de etiquetas de fraude
explícitas.

## El concepto

**Outlier vs anomalía vs novelty.**

- **Outlier:** punto que se desvía del grueso. Puede ser
  ruido, error de medición, o un evento genuino.
- **Anomalía:** outlier con significado (fraude, falla).
- **Novelty:** outlier en datos de test, no de training. El
  modelo aprende de training "limpio" y detecta desviaciones.

**Métodos basados en distribución.** Asumen que los datos
normales siguen una distribución y los outliers caen en las
colas.

- **z-score:** `(x - μ) / σ`. Outlier si `|z| > 3`. Funciona
  con datos gaussianos.
- **IQR:** outlier si `x < Q1 - 1.5·IQR` o `x > Q3 + 1.5·IQR`.
  Robusto a outliers en la propia definición.
- **Mahalanobis:** distancia al centroide considerando
  covarianza. Útil en multidimensional.

**Isolation Forest.** Aísla outliers con árboles de
decisión aleatorios. Los outliers son más fáciles de aislar
(fewer splits) que los puntos normales. El score de anomalía
es el promedio de la profundidad de aislamiento sobre
muchos árboles. No necesita distancia, escala a alta
dimensión.

**Local Outlier Factor (LOF).** Mide la densidad local de un
punto comparada con la de sus vecinos. Un punto con densidad
mucho menor que la de sus vecinos es outlier. Detecta
anomalías **contextuales** (puntos normales en abstracto pero
raros en su vecindario).

**One-class SVM.** Encuentra un hiperplano que separa los
datos normales del origen. Los puntos en el lado equivocado
del hiperplano son anomalías. Útil cuando tienes pocos o
ningún ejemplo de anomalías.

**Autoencoders para anomalías.** Entrena un autoencoder
solo con datos normales. Las anomalías tienen alto error de
reconstrucción. Umbral sobre el MSE de reconstrucción.

**Cuándo usar cada método.**

| Método | Cuándo |
|---|---|
| z-score, IQR | Univariado, distribución conocida |
| Isolation Forest | Multivariado, escalable, primera opción |
| LOF | Anomalías contextuales, baja dimensionalidad |
| One-class SVM | Pocos datos, anomalías raras |
| Autoencoder | Datos complejos (imagen, secuencia) |

**Métricas de evaluación.** La accuracy falla con clases tan
desbalanceadas. Usa:

- **Precision@k:** de los top-k alertas, cuántas son
  anomalías reales. Es la métrica de negocio: si tu equipo
  revisa 100 alertas, cuántas son reales.
- **Recall@k:** de las anomalías reales, cuántas están en
  los top-k.
- **PR-AUC:** área bajo la curva precision-recall. Más
  informativa que ROC con desbalance extremo.

**Trampas.**

- **Umbral muy bajo:** muchas falsas alarmas, el equipo
  ignora las alertas.
- **Umbral muy alto:** pierdes anomalías reales.
- **No monitorear drift:** la distribución de los datos
  cambia, el modelo queda obsoleto.
- **Sin etiquetas de validación:** es tentador usar accuracy
  sobre datos sin etiquetar. No funciona.

## Constrúyelo

```python
import numpy as np


def z_score_outliers(x, threshold=3.0):
    """Outliers por z-score. Asume distribución aproximadamente
    gaussiana."""
    mu = x.mean()
    sigma = x.std() + 1e-9
    return np.abs((x - mu) / sigma) > threshold


def iqr_outliers(x, k=1.5):
    """Outliers por rango intercuartil."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def isolation_score(x, n_trees=100, max_samples=256, semilla=0):
    """Score de anomalía estilo Isolation Forest (simplificado).
    Devuelve score alto para outliers."""
    rng = np.random.default_rng(semilla)
    n = len(x)
    depths = np.zeros(n)
    for _ in range(n_trees):
        idx = rng.choice(n, size=min(max_samples, n), replace=False)
        sample = x[idx]
        # profundidad de aislamiento promedio
        for i_local, i_global in enumerate(idx):
            depths[i_global] += _path_length(x[i_local], sample, rng)
    # normalizar (cálculo aproximado)
    c = _c_factor(min(max_samples, n))
    return 2 ** (-depths / (n_trees * c))


def _path_length(x, sample, rng, current_depth=0):
    """Recorre un árbol de aislamiento aleatorio hasta aislar x."""
    if len(sample) <= 1 or current_depth >= 20:
        return current_depth
    q_min, q_max = sample.min(), sample.max()
    if q_min == q_max:
        return current_depth
    split = rng.uniform(q_min, q_max)
    left = sample[sample < split]
    right = sample[sample >= split]
    if x < split:
        return _path_length(x, left, rng, current_depth + 1)
    return _path_length(x, right, rng, current_depth + 1)


def _c_factor(n):
    """Factor de normalización para Isolation Forest."""
    if n <= 1:
        return 1
    from math import log
    return 2 * (log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


def precision_at_k(y_true, scores, k=100):
    """precision@k: de los top-k scores, cuántos son positivos."""
    top_k_idx = np.argsort(scores)[-k:]
    return float(y_true[top_k_idx].sum() / k)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-anomaly-detection
fase: 02
leccion: 16
---

Eres un asistente que ayuda a configurar detección de anoma-
lías. Recibirás la descripción del problema (tipo de datos,
volumen, presupuesto para falsos positivos, disponibilidad de
etiquetas). Tu trabajo:

1. Univariado y gaussiano: z-score.
2. Univariado con outliers: IQR.
3. Multivariado y escalable: Isolation Forest.
4. Anomalías contextuales: LOF.
5. Datos complejos (imagen, secuencia): autoencoder.
6. Si hay etiquetas de anomalías (incluso pocas): ajustar
   umbral con precision@k.
7. Monitorear drift: reentrenar el detector cuando la
   distribución cambie.
8. Reportar precision@k, recall@k y PR-AUC.
9. Advertir contra accuracy con clases desbalanceadas.
```

## Ejercicios

1. **Isolation Forest desde cero**: implementa IForest
   simplificado y aplica a un dataset sintético con
   anomalías conocidas.
2. **LOF**: aplica LOF a un dataset con clusters de
   densidad variable.
3. **Desafío**: implementa detección de anomalías con
   autoencoder en MNIST y detecta dígitos "raros".

## Lecturas recomendadas

- *Outlier Analysis* — Aggarwal.
- Isolation Forest paper: Liu, Ting, Zhou, 2008.
- scikit-learn anomaly: <https://scikit-learn.org/stable/modules/outlier_detection.html>.
- PyOD: <https://pyod.readthedocs.io>.

---

> 📚 **Adaptación al español** de la lección "[Anomaly Detection]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
