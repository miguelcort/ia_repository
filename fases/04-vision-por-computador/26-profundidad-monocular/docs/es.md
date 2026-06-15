# 26 — Profundidad monocular y geometría

> Estimar la profundidad de cada píxel desde una sola imagen. Es la base de efectos bokeh, AR, robótica, y mapas 3D.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-segmentacion-semantica-unet,
                  18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un modelo de monocular depth estimation.
- Aplicar MiDaS, Depth Anything, o DPT.
- Diagnosticar métricas de depth: AbsRel, RMSE, δ-accuracy.
- Conocer SfM, NeRF, y 3DGS como alternativas.

## El problema

Una cámara 3D produce depth maps con sensores como LiDAR o
time-of-flight. Una cámara monocular (tu celular) solo
produce RGB. Estimar la profundidad por píxel desde una
sola imagen es un problema mal puesto (monocular depth
cues son ambiguos), pero los modelos de deep learning
producen estimaciones útiles. La lección cubre los
métodos canónicos: DPT, MiDaS, Depth Anything, y la
evaluación.

## El concepto

**DPT (Vision Transformers for Dense Prediction, Ranftl
et al., 2021).** Aplica ViT como encoder y decoder
denso con regresión pixel-wise. Preentrenado en
mezcla de datasets (Mix, DIML, etc.). SOTA en 2021.

**MiDaS v3 (Ranftl et al., 2022).** Mezcla de 10+ datasets
de depth. Usa Vision Transformer como backbone. Modelo
zero-shot: predice depth relativo que puede convertirse a
métrico con un shift/scale aprendido en el dataset target.
SOTA en monocular depth.

**Depth Anything (Yang et al., 2024).** Mejora MiDaS
con un teacher-student scheme y 62M de unlabeled
images. SOTA en 2024. Disponible en Hugging Face.

**Depth Anything v2 (2024).** Mejora aún más usando
synthetic data de teacher models. SOTA en 2024-2025.

**Métricas de evaluación.**

- **AbsRel:** `mean(|pred - gt| / gt)`. Error relativo
  promedio.
- **RMSE:** `sqrt(mean((pred - gt)²))`. Error cuadrático.
- **δ < 1.25:** fracción de píxeles donde
  `max(pred/gt, gt/pred) < 1.25`. Accuracy threshold.
- **SILog:** scale-invariant log error. Común en KITTI.

**Scale ambiguity.** Monocular depth no conoce la escala
absoluta: una foto de cerca de un juguete vs de lejos de
un edificio pueden tener la misma apariencia relativa.
Para depth en metros, necesitas calibración con un sensor
(LiDAR, GPS, IMU) o asumir un objeto de tamaño conocido.

**Aplicaciones.**

- **Efecto bokeh en retratos:** enfocar al sujeto y
  desenfocar el fondo.
- **AR/VR:** colocar objetos virtuales en el mundo real.
- **Robótica:** navegación y manipulación.
- **Mapas 3D:** convertir fotos en mapas.
- **Editing:** inpainting consciente de profundidad.

**Trampas.**

- **Sin ground truth métrico:** los datasets usan
  LiDAR o SfM, que tienen su propio error. AbsRel < 0.1
  es SOTA.
- **Cámaras móviles:** la calibración cambia. Recalibrar
  o usar depth relativo.
- **Escala incorrecta:** verificar que el depth predicho
  está en la escala correcta antes de usarlo.

## Constrúyelo

```python
import numpy as np


def abs_rel(pred, gt, mask=None):
    """AbsRel: mean(|pred - gt| / gt)."""
    if mask is None:
        mask = gt > 0
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def rmse(pred, gt, mask=None):
    if mask is None:
        mask = gt > 0
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def delta_accuracy(pred, gt, threshold=1.25, mask=None):
    if mask is None:
        mask = gt > 0
    ratio = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return float(np.mean(ratio < threshold))


def silog(pred, gt, mask=None):
    """Scale-invariant log error."""
    if mask is None:
        mask = gt > 0
    log_diff = np.log(pred[mask]) - np.log(gt[mask])
    return float(np.mean(log_diff ** 2) - 0.5 * np.mean(log_diff) ** 2)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-depth
fase: 04
leccion: 26
---

Eres un asistente que ayuda a estimar profundidad monocu-
lar. Recibirás las imágenes y la aplicación. Tu trabajo:

1. Si quieres SOTA: Depth Anything v2 de Hugging Face.
2. Si quieres velocidad: MiDaS small.
3. Si quieres accuracy absoluta: DPT-Large con
   calibration.
4. Para AR/VR: usar depth métrico con scale recovery
   (asumir un objeto de tamaño conocido).
5. Para bokeh: depth relativo es suficiente.
6. Métricas: AbsRel, RMSE, δ<1.25, SILog.
7. Preprocesar: image a RGB, resize al input del modelo
   (e.g. 518x518 para DPT).
8. Postprocesar: smooth, inpaint holes, edge-aware
   filter.
```

## Ejercicios

1. **Depth Anything**: aplica Depth Anything v2 a un
   dataset de fotos.
2. **Métricas**: implementa AbsRel, RMSE, δ-accuracy
   sobre un dataset con GT.
3. **Desafío**: entrena un modelo de depth desde cero
   con un dataset pequeño.

## Lecturas recomendadas

- *Vision Transformers for Dense Prediction* — Ranftl et
  al., 2021.
- *Towards Robust Monocular Depth Estimation* — Ranftl
  et al., 2022 (MiDaS).
- *Depth Anything* — Yang et al., 2024.
- *Depth Anything v2* — Yang et al., 2024.
- Hugging Face transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Monocular Depth and Geometry]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
