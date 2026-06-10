"""
Lección: 08-segmentacion-de-instancia-mask-rcnn
Fase: 04
Prerrequisitos: 07-segmentacion-semantica-unet
"""
from __future__ import annotations
import sys
import numpy as np


def masks_to_boxes(mascaras):
    """Convierte lista de mascaras binarias (H, W) a bboxes [x1, y1, x2, y2]."""
    boxes = []
    for m in mascaras:
        if m.sum() == 0:
            boxes.append([0, 0, 0, 0])
            continue
        ys, xs = np.where(m > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes.append([int(x1), int(y1), int(x2) + 1, int(y2) + 1])
    return boxes


def mask_iou(mask_a, mask_b):
    """IoU entre dos mascaras binarias."""
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def panoptic_quality(pred_seg, pred_inst, gt_seg, gt_inst):
    """Panoptic quality: PQ = SQ * DQ.
    SQ: average IoU of matched segments. DQ: precision * recall de matches."""
    # Simplificado: PQ = sum(IoU_tp) / (|TP| + 0.5|FP| + 0.5|FN|)
    if not isinstance(pred_seg, np.ndarray):
        pred_seg = np.array(pred_seg)
    return None  # implementación completa requiere matching hungaro


def format_coco_segmentation(boxes, mascaras, clases, scores=None):
    """Convierte resultados a formato COCO: lista de {bbox, segmentation, category_id, score}."""
    resultados = []
    for i, (b, m) in enumerate(zip(boxes, mascaras)):
        entry = {
            "bbox": b,
            "segmentation": m.astype(np.uint8).tolist(),
            "category_id": clases[i] if i < len(clases) else 0,
        }
        if scores is not None and i < len(scores):
            entry["score"] = float(scores[i])
        resultados.append(entry)
    return resultados


def main() -> int:
    mascara1 = np.zeros((10, 10), dtype=bool)
    mascara1[2:5, 2:5] = True
    mascara2 = np.zeros((10, 10), dtype=bool)
    mascara2[6:8, 6:8] = True
    boxes = masks_to_boxes([mascara1, mascara2])
    print(f"Boxes: {boxes}")
    print(f"Mask IoU (1, 2): {mask_iou(mascara1, mascara2):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())