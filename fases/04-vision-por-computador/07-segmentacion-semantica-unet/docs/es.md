# Segmentación semántica con U-Net

> Asignar una clase a cada pixel. U-Net (Ronneberger, 2015) sigue siendo la base: encoder-decoder con skip connections que preservan detalle espacial. La misma forma de Stable Diffusion.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar down/upsample.
- Implementar Dice loss e IoU.
- Entender la arquitectura encoder-decoder con skip.

## Constrúyelo

```python
def dice_loss(y_true, y_pred, eps=1e-7):
    intersection = (y_true * y_pred).sum()
    return 1 - (2 * intersection + eps) / (y_true.sum() + y_pred.sum() + eps)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-segmentation
fase: 04
leccion: 07
---

1. Medical: U-Net 3D + Dice, ventanas HU.
2. Autonomia: DeepLabV3+, SegFormer.
3. Satelite: U-Net + EfficientNet.
4. Few-shot: SAM.
5. BCE + Dice loss, mIoU + Dice metricas.
```

## Ejercicios

1. **Tversky loss**: variante de Dice con pesos FP/FN.
2. **Boundary loss**: optimiza precision de bordes.
3. **Desafio**: implementar U-Net completo y entrenar en
   un dataset de medical (e.g. BUSI o ISIC).

## Lecturas recomendadas

- "U-Net" (Ronneberger et al., 2015)
- "V-Net" (Milletari et al., 2016)
- nnU-Net: <https://github.com/MIC-DKFZ/nnUNet>

---

> 📚 **Adaptación al español** de la lección "[Semantic Segmentation with U-Net]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).