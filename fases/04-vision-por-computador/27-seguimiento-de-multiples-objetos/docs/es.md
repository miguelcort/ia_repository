# 27 — Tracking multi-objeto y memoria de video

> Seguir múltiples objetos a lo largo de frames en video. La base de vigilancia, análisis deportivo, robots, y AR.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-deteccion-de-objetos-yolo,
                  12-comprension-de-video
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar SORT y ByteTrack.
- Aplicar tracking con detección + asociación.
- Diagnosticar ID switches y fragmentación.
- Conocer tracking moderno: BoT-SORT, Deep OC-SORT.

## El problema

La detección te dice "hay un auto en este frame, en esta
posición". El **tracking** te dice "este auto es el mismo
que estaba en el frame anterior, ahora se movió aquí". Sin
tracking, cada frame es independiente: no hay identidad, no
hay trayectoria. La lección cubre SORT, ByteTrack, y las
métricas para evaluar.

## El concepto

**SORT (Simple Online and Realtime Tracking, Bewley et
al., 2016).** Algoritmo clásico:

1. Detectar objetos en cada frame.
2. Predecir la nueva posición con Kalman filter.
3. Asociar detecciones con tracks existentes usando
   Hungarian algorithm sobre IoU.
4. Tracks sin match: nuevos. Detecciones sin match:
   perdidos.

Es simple y rápido pero falla con oclusiones.

**ByteTrack (Zhang et al., 2022).** Mejora SORT usando
**todas** las detecciones, no solo las de alta confidence.
Asociar primero las de alta confidence, luego asociar las
de baja confidence con los tracks restantes. Esto
recupera tracks ocluidos con baja confidence.

**BoT-SORT (Aharon et al., 2022).** Añade apariencia
(embeddings visuales) y mejora el Kalman filter con
camera motion compensation. SOTA en MOT benchmarks.

**Deep OC-SORT (Cao et al., 2023).** Más robusto a
movimientos no lineales y oclusiones prolongadas.

**Kalman filter.** Modelo lineal gaussiano del estado del
objeto (posición + velocidad). Predice la siguiente
posición; al recibir una medición (detección), actualiza.
Es el "predictor" en SORT/ByteTrack.

**Hungarian algorithm.** Resuelve la asignación óptima
entre N tracks y M detecciones minimizando un costo
(usualmente `-IoU` o distancia). Es `O(N³)` pero
manejable para N pequeño.

**Métricas de tracking.**

- **MOTA (Multiple Object Tracking Accuracy):**
  `1 - (FN + FP + IDsw) / GT`. Mide errores totales.
- **MOTP (Multiple Object Tracking Precision):**
  distancia promedio entre predicción y GT. Mide
  precisión de la posición.
- **ID switches:** número de veces que un track cambia
  de ID. Debe ser bajo.
- **IDF1:** F1 entre ground-truth IDs y predicted IDs.
  Mide consistencia.

**Trampas.**

- **Movimiento no lineal:** el Kalman filter asume
  movimiento lineal. Usar BoT-SORT con apariencia o
  deep OC-SORT.
- **Oclusión prolongada:** el track se pierde. Usar
  memoria de apariencia (ReID embeddings).
- **Muchas detecciones falsas:** pre-filtrar con
  confidence threshold.

## Constrúyelo

```python
import numpy as np


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-12)


def kalman_predict(state, dt=1.0):
    """Predice el siguiente estado. state: [x, y, w, h, vx, vy]."""
    pred = state.copy()
    pred[0] += state[4] * dt
    pred[1] += state[5] * dt
    return pred


def kalman_update(state, measurement):
    """Actualiza el estado con una medición (x1, y1, x2, y2)."""
    measured = np.array([
        (measurement[0] + measurement[2]) / 2,
        (measurement[1] + measurement[3]) / 2,
        measurement[2] - measurement[0],
        measurement[3] - measurement[1],
    ])
    return 0.5 * state[:4] + 0.5 * measured  # simplificado


def sort_step(tracks, detections, iou_threshold=0.3):
    """Un paso de SORT: asocia detecciones con tracks por IoU."""
    if not tracks or not detections:
        return tracks, list(range(len(detections)))
    # Predecir cada track
    predicted = [kalman_predict(t["state"]) for t in tracks]
    # Calcular matriz de costos
    n_t, n_d = len(predicted), len(detections)
    cost = np.zeros((n_t, n_d))
    for i, p in enumerate(predicted):
        for j, d in enumerate(detections):
            box_p = [p[0] - p[2] / 2, p[1] - p[3] / 2,
                     p[0] + p[2] / 2, p[1] + p[3] / 2]
            cost[i, j] = 1 - iou(box_p, d["box"])
    # Hungarian algorithm (asignación de costo mínimo)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # Aceptar matches con IoU > threshold
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1 - iou_threshold:
            matches.append((r, c))
    unmatched_d = [j for j in range(n_d)
                    if j not in {c for _, c in matches}]
    unmatched_t = [i for i in range(n_t)
                    if i not in {r for r, _ in matches}]
    return matches, unmatched_d, unmatched_t
```

## Úsalo

```bash
pip install filterpy scipy
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tracking
fase: 04
leccion: 27
---

Eres un asistente que ayuda a implementar tracking
multi-objeto. Recibirás el video, el detector, y los
requisitos. Tu trabajo:

1. Si quieres velocidad: SORT.
2. Si quieres accuracy: ByteTrack o BoT-SORT.
3. Si hay oclusiones prolongadas: Deep OC-SORT.
4. Detección: YOLOv8n/m o RT-DETR.
5. Asociación: Hungarian algorithm sobre IoU
   (motion) + cosine similarity (apariencia).
6. Kalman filter: predicción lineal.
7. Métricas: MOTA, MOTP, IDF1, ID switches.
8. Para ReID: usar OSNet o FastReID para embeddings
   visuales.
9. Visualizar: dibujar cada track con un color
   consistente.
```

## Ejercicios

1. **SORT**: implementa SORT completo y aplícalo a un
   video de prueba.
2. **ByteTrack**: implementa ByteTrack y compara con
   SORT en un video con oclusiones.
3. **Desafío**: implementa BoT-SORT con embeddings
   visuales.

## Lecturas recomendadas

- *Simple Online and Realtime Tracking (SORT)* —
  Bewley et al., 2016.
- *ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box* — Zhang et al., 2022.
- *BoT-SORT* — Aharon et al., 2022.
- *Deep OC-SORT* — Cao et al., 2023.
- BoxMOT: <https://github.com/mikel-brostrom/boxmot>.

---

> 📚 **Adaptación al español** de la lección "[Multi-Object Tracking]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
