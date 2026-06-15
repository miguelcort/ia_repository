# 21 — Puntos clave y estimación de poses

> Detectar keypoints (esquinas, articulaciones, landmarks) es la base de pose estimation, face landmark, y muchas aplicaciones de interacción.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-deteccion-de-objetos-yolo
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar detección de keypoints con heatmaps.
- Aplicar pose estimation (humana, mano, cara).
- Diagnosticar errores comunes en multi-persona.
- Conocer OpenPose, MediaPipe, MMPose.

## El problema

La detección dice "dónde hay una persona" con una bounding
box. La **estimación de poses** dice "dónde están sus
articulaciones" — muñecas, codos, hombros, caderas,
rodillas, tobillos. Es la base de análisis deportivo,
animación, control de gestos, y AR/VR. La lección cubre
los métodos canónicos: heatmap-based (OpenPose, HRNet) y
regression-based (MediaPipe BlazePose).

## El concepto

**Heatmap-based keypoint detection.** Para cada
keypoint `k`, la red produce un heatmap `(H, W)` donde
cada píxel tiene un score igual a la probabilidad de que
el keypoint esté en esa posición. El argmax del heatmap
da la posición predicha. La ground truth se genera con
un Gaussian centrado en la posición real.

**Stacked hourglass (Newell et al., 2016).** Encoder-
decoder con skip connections. Aplica "hourglass modules"
repetidamente para refinar predicciones. Base de OpenPose.

**HRNet (Wang et al., 2020).** Mantiene representaciones
de múltiples resoluciones en paralelo (no encoder-decoder
secuencial). Conexiones entre resoluciones para fusionar
info. SOTA en COCO keypoints.

**OpenPose (Cao et al., 2017).** Usa Part Affinity
Fields (PAF): campos vectoriales 2D que codifican la
asociación entre keypoints (e.g. que este codo
pertenece a esta muñeca). Permite multi-persona sin
conocer el número de personas a priori.

**MediaPipe BlazePose.** Regression-based: predice
directamente las coordenadas 3D de 33 keypoints. Más
rápido que heatmap, ideal para móviles.

**Métricas.**

- **PCK (Percentage of Correct Keypoints):** keypoint
  predicho está dentro de un threshold del ground truth.
- **OKS-based mAP (COCO):** Object Keypoint Similarity.
  Promedio sobre thresholds.
- **MPJPE (Mean Per Joint Position Error):** error
  promedio por articulación en mm, usado en 3D pose.

**Datasets.**

- **COCO keypoints:** 17 keypoints, 250k personas.
- **MPII Human Pose:** 16 keypoints, 40k imágenes.
- **Human3.6M:** 3D pose, 3.6M frames.

**Trampas.**

- **Multi-persona sin associations:** predecir
  keypoints de todas las personas lleva a confusión.
  Usar OpenPose PAF o instancia segmentation primero.
- **Heatmap resolution:** heatmaps a baja resolución
  pierden precisión. Mantener heatmaps a alta
  resolución y luego downsampling.
- **Self-occlusion:** keypoints ocluidos son ambiguos.
  El modelo puede predecir cualquier posición
  plausible.

## Constrúyelo

```python
import numpy as np


def generate_heatmap(keypoint, H, W, sigma=2.0):
    """Genera un heatmap gaussiano centrado en keypoint (x, y)."""
    x, y = keypoint
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-d2 / (2 * sigma ** 2))


def decode_heatmap(heatmap):
    """Decodifica el argmax + refinamiento subpixel."""
    idx = np.argmax(heatmap)
    y, x = np.unravel_index(idx, heatmap.shape)
    # Refinamiento subpixel
    h, w = heatmap.shape
    if 0 < y < h - 1 and 0 < x < w - 1:
        dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
        dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
        return x + dx, y + dy
    return x, y


def pck(pred_keypoints, gt_keypoints, threshold=0.1, H=64, W=64):
    """PCK: keypoint dentro de threshold de gt."""
    dists = np.sqrt(np.sum(
        (np.array(pred_keypoints) - np.array(gt_keypoints)) ** 2, axis=-1
    ))
    return float(np.mean(dists < threshold * max(H, W)))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-keypoints
fase: 04
leccion: 21
---

Eres un asistente que ayuda a implementar detección de
keypoints o pose estimation. Recibirás la tarea, número
de personas, y hardware. Tu trabajo:

1. Si multi-persona: OpenPose con PAF o HRNet + top-down.
2. Si single-persona y velocidad: MediaPipe BlazePose.
3. Si quieres SOTA: HRNet o ViTPose preentrenado en COCO.
4. Si 3D: VideoPose3D o METRO.
5. Métricas: PCK@0.1, OKS mAP (COCO), MPJPE (3D).
6. Augmentations: RandomCrop, RandomRotation, ColorJitter.
7. Preentrenar en COCO, fine-tune en tu dominio.
8. Para heatmap-based: ground truth con Gaussian de
   sigma=2.
```

## Ejercicios

1. **Heatmaps**: implementa generación y decodificación
   de heatmaps.
2. **Pose estimation**: aplica MediaPipe a un video y
   visualiza los keypoints.
3. **Desafío**: implementa un pipeline de pose estimation
   en tiempo real con webcam.

## Lecturas recomendadas

- *OpenPose: Realtime Multi-Person 2D Pose Estimation* —
  Cao et al., 2017.
- *HRNet* — Wang et al., 2020.
- *MediaPipe BlazePose* — Bazarevsky et al., 2020.
- MMPose: <https://github.com/open-mmlab/mmpose>.

---

> 📚 **Adaptación al español** de la lección "[Keypoints and Pose Estimation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
