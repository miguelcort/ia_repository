# 06 — Detección de objetos: YOLO

> YOLO detecta y localiza múltiples objetos en una imagen en tiempo real. Es la opción dominante para aplicaciones prácticas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar los componentes de YOLO: anchor boxes, IoU,
  NMS.
- Entrenar YOLOv8 en un dataset de detección.
- Calcular mAP (mean Average Precision) sobre test.
- Diagnosticar trade-offs velocidad/accuracy.

## El problema

Clasificar dice "qué hay" en la imagen. Detectar dice
"qué hay **y dónde**". Esto requiere predecir bounding
boxes (x, y, width, height) más una clase. YOLO (You Only
Look Once) reformuló la detección como un único problema
de regresión: la red produce todas las cajas en una sola
pasada forward. Por eso es rápido: 30-150 FPS en GPU.

## El concepto

**Bounding boxes.** Cada objeto se representa como `(x, y,
w, h, class, confidence)`. `x, y` es el centro, `w, h` el
tamaño. `confidence` es la probabilidad de que haya un objeto
en esa caja.

**Anchor boxes.** Cajas预definidas de ciertas proporciones
(1:1, 1:2, 2:1, etc.). YOLO predice **offsets** respecto a
estos anchors, no cajas absolutas. Esto facilita el
aprendizaje: la red solo necesita aprender cuánto
desviarse del anchor.

**IoU (Intersection over Union).** `IoU = area_interseccion /
area_union`. Mide el solapamiento entre dos cajas. Es la
métrica base para matching y NMS.

**NMS (Non-Maximum Suppression).** Elimina cajas
redundantes. Algoritmo:

1. Ordenar cajas por confidence descendente.
2. Seleccionar la caja con mayor confidence.
3. Eliminar todas las cajas con IoU > threshold (típico
   0.5) con la seleccionada.
4. Repetir hasta que no queden cajas.

**Arquitectura YOLOv8.** Backbone CSPDarknet, neck PANet,
head decoupled. Tres escalas de detección (P3, P4, P5) para
objetos de diferentes tamaños. YOLOv8n (nano) corre en
edge; YOLOv8x (extra-large) alcanza SOTA accuracy.

**mAP (mean Average Precision).** Métrica estándar en
detección. Para cada clase, calcular el área bajo la curva
precision-recall. Promediar sobre todas las clases. mAP@0.5
usa IoU threshold 0.5; mAP@0.5:0.95 promedia sobre
thresholds de 0.5 a 0.95.

**Datasets canónicos.** COCO (80 clases, 118k imágenes),
Pascal VOC (20 clases), Cityscapes (autos), BDD100k.

**Pipeline de entrenamiento.**

1. Cargar imágenes con bounding boxes.
2. Data augmentation: mosaic, mixup, random affine,
   color jitter.
3. Asignar cajas a escalas (los objetos grandes van a
   head P5, pequeños a P3).
4. Loss = box_loss + cls_loss + obj_loss (YOLO loss).
5. Entrenar ~300 epochs con SGD o AdamW.

**Trampas.**

- **NMS muy agresivo:** pierde detecciones válidas en
  objetos solapados.
- **Anchor boxes inadecuadas:** la red lucha por
  predecir offsets grandes. Usa K-means sobre tus cajas
  para elegir mejores anchors.
- **Confidence threshold muy bajo:** muchas detecciones
  falsas.

## Constrúyelo

```python
import numpy as np


def iou(box1, box2):
    """IoU entre dos cajas [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-12)


def nms(boxes, scores, threshold=0.5):
    """Non-maximum suppression. boxes: (N, 4) en formato
    [x1, y1, x2, y2]."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= threshold]
    return keep


def xywh_a_xyxy(box):
    """Convierte [x_center, y_center, w, h] a [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-yolo
fase: 04
leccion: 06
---

Eres un asistente que ayuda a entrenar un detector de
objetos. Recibirás el dataset (n, número de clases), el
hardware y la latencia objetivo. Tu trabajo:

1. Si quieres velocidad: YOLOv8n o YOLOv8s.
2. Si quieres accuracy: YOLOv8m o YOLOv8l.
3. Si quieres balance: YOLOv8x.
4. Para edge / móvil: YOLOv8n con quantization.
5. Preentrenado en COCO, fine-tune en tu dataset.
6. Augmentations: mosaic, mixup, color jitter.
7. Entrenar 300 epochs con SGD o AdamW.
8. Evaluar con mAP@0.5 y mAP@0.5:0.95.
9. NMS con threshold 0.5; bajar si muchos solapados.
10. Confidence threshold: 0.25 default; ajustar según
    tasa de falsos positivos.
```

## Ejercicios

1. **NMS**: implementa NMS y visualiza antes/después.
2. **IoU**: implementa IoU y mAP sobre un dataset
   pequeño.
3. **Desafío**: entrena YOLOv8n en un dataset de
   detección de autos y mide mAP.

## Lecturas recomendadas

- *You Only Look Once* — Redmon et al., 2016.
- *YOLOv8* — Ultralytics docs: <https://docs.ultralytics.com>.
- *COCO dataset*: <https://cocodataset.org>.
- mAP: <https://github.com/rafaelpadilla/Object-Detection-Metrics>.

---

> 📚 **Adaptación al español** de la lección "[Object Detection: YOLO]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
