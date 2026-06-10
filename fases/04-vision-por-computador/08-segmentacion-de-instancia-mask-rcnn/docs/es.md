# Segmentación de instancia con Mask R-CNN

> Cada perro, cada persona, cada coche: su propia mascara. Mask R-CNN anade una cabeza de mascara a Faster R-CNN, y sigue siendo el standard de precision para clases custom.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-segmentacion-semantica-unet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Convertir mascaras a bboxes.
- Calcular IoU entre mascaras.
- Formatear resultados en COCO JSON.
- Diagnosticar instance vs semantic vs panoptic.

## Constrúyelo

```python
def masks_to_boxes(mascaras):
    boxes = []
    for m in mascaras:
        if m.sum() == 0:
            boxes.append([0, 0, 0, 0])
            continue
        ys, xs = np.where(m > 0)
        boxes.append([int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-instance-seg
fase: 04
leccion: 08
---

1. Precision custom: Mask R-CNN ResNet-50 FPN.
2. Tiempo real: YOLACT o YOLOv8-seg.
3. Zero-shot: SAM.
4. Medical: Cellpose, Stardist.
5. mAP@[.5:.95] como metrica.
```

## Ejercicios

1. **Mask NMS**: NMS sobre mascaras con mask IoU en vez
   de box IoU.
2. **COCO eval**: implementa mAP@[.5:.95] con COCO API.
3. **Desafio**: entrenar Mask R-CNN en un dataset custom
   (e.g. deteccion de nucleos con BBBC038).

## Lecturas recomendadas

- "Mask R-CNN" (He et al., 2017)
- "YOLACT" (Bolya et al., 2019)
- "Segment Anything" (Kirillov et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Instance Segmentation with Mask R-CNN]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).