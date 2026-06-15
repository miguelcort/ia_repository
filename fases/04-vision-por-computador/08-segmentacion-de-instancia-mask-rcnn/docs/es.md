# 08 — Segmentación de instancia: Mask R-CNN

> Mask R-CNN detecta, clasifica, y segmenta cada instancia individual. La base de muchos sistemas de visión industrial.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-deteccion-de-objetos-yolo,
                  07-segmentacion-semantica-unet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la arquitectura Mask R-CNN.
- Entrenar con datasets como COCO.
- Calcular métricas de segmentación de instancia.
- Diagnosticar trade-offs vs YOLO + segmentación.

## El problema

La segmentación semántica no distingue entre dos autos
solapados: ambos son píxeles "auto". La **segmentación de
instancia** asigna una máscara a cada instancia individual.
Es lo que necesitas para contar objetos,跟踪 trayectorias,
o editar uno sin afectar al otro. Mask R-CNN (He et al.,
2017) extiende Faster R-CNN agregando una rama de
segmentación que predice la máscara para cada RoI.

## El concepto

**Mask R-CNN.** Tres componentes:

1. **Backbone (ResNet + FPN):** extrae features.
2. **RPN (Region Proposal Network):** propone regiones
   candidatas donde podría haber objetos.
3. **Cabecera dual:** clasifica la región y refina la
   caja + predice la máscara binaria (K clases, una por
   cada una).

**RoIAlign.** En vez de RoIPool (que cuantiza y pierde
precisión), RoIAlign usa interpolación bilineal para
mantener la alineación de píxeles. Crítico para máscaras
precisas.

**FPN (Feature Pyramid Network).** Combina features de
múltiples escalas del backbone. Los objetos pequeños se
detectan en capas de alta resolución; los grandes en capas
de baja resolución.

**Pérdida multi-task.** `L = L_cls + L_box + L_mask`. La
pérdida de máscara es binary cross-entropy per-pixel,
solo sobre la región RoI.

**Training pipeline.**

1. Pre-entrenar el backbone en ImageNet.
2. Entrenar RPN + cabeza con RPN proposals.
3. Alternar optimización entre RPN y cabeza.

**Métricas.**

- **mAP@[.5, .75, .5:.95]:** standard de COCO.
- **AP por clase:** útil para identificar clases difíciles.
- **Mask mAP:** AP sobre la máscara, separado de box AP.

**Cuándo usar Mask R-CNN vs YOLO + segmentación.**

| Aspecto | Mask R-CNN | YOLO + mask head |
|---|---|---|
| Velocidad | Lento (~5 FPS) | Rápido (50+ FPS) |
| Accuracy | Alta | Media-alta |
| Dificultad de training | Media | Media |
| Mejor para | Calidad máxima | Tiempo real |

**Alternativas modernas.**

- **YOLOv8-seg:** segmentación de instancia rápida, ~30 FPS.
- **Mask2Former:** universal (semántica, instancia,
  panóptica) con transformer.
- **SAM (Segment Anything):** promptable, zero-shot,
  base de muchas aplicaciones.

**Trampas.**

- **RPN demasiado agresivo:** muchas proposals, cabeza
  lenta. Limitar a top-1000 proposals.
- **RoIAlign con tamaño mal elegido:** máscaras
  pixeladas. Default 28x28 para la cabeza de máscara.
- **Threshold de máscara muy bajo:** mucho ruido. Usar
  0.5 o ajustar por clase.

## Constrúyelo

```python
import numpy as np


def mask_iou(mask1, mask2):
    """IoU entre dos máscaras binarias del mismo tamaño."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / max(union, 1e-12)


def mask_ap(pred_masks, pred_scores, pred_classes,
            gt_masks, gt_classes, iou_threshold=0.5):
    """Average Precision para máscaras, similar a detección."""
    # Ordenar predicciones por score descendente
    order = np.argsort(-pred_scores)
    pred_classes = pred_classes[order]
    pred_masks = pred_masks[order]

    n_pred = len(pred_masks)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    for i in range(n_pred):
        # Buscar GT del mismo clase con mayor IoU
        best_iou = 0
        for j, gt_c in enumerate(gt_classes):
            if gt_c != pred_classes[i]:
                continue
            iou = mask_iou(pred_masks[i], gt_masks[j])
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_threshold:
            tp[i] = 1
        else:
            fp[i] = 1

    # Acumuladas y precision-recall curve
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(len(gt_masks), 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # AP = área bajo la curva PR
    ap = 0
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return ap
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mask-rcnn
fase: 04
leccion: 08
---

Eres un asistente que ayuda a entrenar Mask R-CNN. Reci-
birás el dataset, la latencia objetivo, y el hardware. Tu
trabajo:

1. Si quieres accuracy máxima: Mask R-CNN con ResNet-101
   + FPN preentrenado en COCO.
2. Si quieres velocidad: YOLOv8-seg o Mask2Former.
3. Preentrenar en COCO, fine-tune en tu dataset.
4. Augmentations: mosaic, mixup, color jitter.
5. Entrenar con SGD, lr=1e-3, 12 epochs.
6. Métrica: mAP@[.5, .75, .5:.95] sobre máscara.
7. NMS con threshold 0.5.
8. Confidence threshold 0.7 para producción.
9. Mask threshold 0.5 para binarizar.
```

## Ejercicios

1. **Mask IoU**: implementa mask_iou y visualiza
   match/mismatch entre predicciones y ground truth.
2. **Mask AP**: implementa el cálculo de mAP sobre
   máscaras.
3. **Desafío**: entrena Mask R-CNN en un dataset
   pequeño y mide mAP.

## Lecturas recomendadas

- *Mask R-CNN* — He et al., 2017.
- *Feature Pyramid Networks* — Lin et al., 2017.
- *Mask2Former* — Cheng et al., 2022.
- detectron2: <https://github.com/facebookresearch/detectron2>.

---

> 📚 **Adaptación al español** de la lección "[Instance Segmentation: Mask R-CNN]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
