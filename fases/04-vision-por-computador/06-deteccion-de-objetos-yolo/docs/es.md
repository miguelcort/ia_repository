# Detección de objetos con YOLO

> Dada una imagen con N objetos, decir que son y donde estan. YOLO lo hace en una sola pasada de la red: rapido (45+ FPS), preciso, y la base de deteccion moderna.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular IoU entre bounding boxes.
- Implementar Non-Maximum Suppression.
- Generar anchor boxes.
- Diagnosticar mAP en datasets custom.

## Constrúyelo

```python
def nms(boxes, scores, iou_threshold=0.5):
    orden = np.argsort(scores)[::-1]
    keep = []
    while len(orden) > 0:
        i = orden[0]
        keep.append(int(i))
        if len(orden) == 1:
            break
        ious = np.array([iou(boxes[i], boxes[j]) for j in orden[1:]])
        orden = orden[1:][ious < iou_threshold]
    return keep
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-detect-model
fase: 04
leccion: 06
---

1. Tiempo real: YOLOv8-N/S, RT-DETR.
2. Balance: YOLOv8-M, RT-DETR-R50.
3. Max accuracy: DINO, Co-DETR, YOLOv8-X.
4. Edge: YOLOv8-Nano, MobileDet.
5. ultralytics para empezar rapido.
```

## Ejercicios

1. **Soft-NMS**: en vez de eliminar, bajar el score
   proporcional al IoU.
2. **Anchor-free**: implementa FCOS-style (centros + distancia).
3. **Desafio**: entrenar YOLOv8-Nano en un dataset custom
   (e.g. deteccion de caras con WIDERFACE subset).

## Lecturas recomendadas

- "You Only Look Once" (Redmon et al., 2016)
- "YOLOv8" (Jocher et al., 2023)
- ultralytics: <https://github.com/ultralytics/ultralytics>

---

> 📚 **Adaptación al español** de la lección "[Object Detection with YOLO]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).