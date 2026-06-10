"""
Lección: 06-deteccion-de-objetos-yolo
Fase: 04
Prerrequisitos: 03-cnns-desde-lenet-hasta-resnet
"""
from __future__ import annotations
import sys
import numpy as np


def iou(box_a, box_b):
    """IoU entre dos boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression. Elimina boxes redundantes."""
    if len(boxes) == 0:
        return []
    orden = np.argsort(scores)[::-1]
    keep = []
    while len(orden) > 0:
        i = orden[0]
        keep.append(int(i))
        if len(orden) == 1:
            break
        # IoU con el resto
        ious = np.array([iou(boxes[i], boxes[j]) for j in orden[1:]])
        orden_restante = orden[1:][ious < iou_threshold]
        orden = orden_restante
    return keep


def generar_anchors(base_size=32, ratios=[0.5, 1.0, 2.0], scales=[1.0, 2.0 ** (1/3), 2.0 ** (2/3)]):
    """Genera anchors para una escala base. ratios y scales de YOLO/Faster R-CNN."""
    anchors = []
    for r in ratios:
        for s in scales:
            w = base_size * s * np.sqrt(r)
            h = base_size * s / np.sqrt(r)
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            anchors.append([x1, y1, x2, y2])
    return np.array(anchors)


def xywh_a_xyxy(box):
    """Convierte [cx, cy, w, h] -> [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def main() -> int:
    # Ejemplo NMS
    boxes = [
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # muy solapado con el primero
        [50, 50, 60, 60],  # separado
    ]
    scores = [0.9, 0.85, 0.7]
    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"Indices despues de NMS: {keep}")
    print(f"IoU(0, 1) = {iou(boxes[0], boxes[1]):.3f}")
    anchors = generar_anchors()
    print(f"Num anchors: {len(anchors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())