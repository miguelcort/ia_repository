# Clasificación de imágenes

> El problema clásico de vision: dada una imagen, asignar una clase. ImageNet (1.2M, 1000 clases) es el benchmark; los humanos hacen ~5% error top-5, los modelos actuales ~10% top-5 con ViT.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar softmax + cross-entropy.
- Implementar top-k accuracy y confusion matrix.
- Calcular F1 macro desde confusion matrix.
- Aplicar data augmentation (flip, random crop).

## Constrúyelo

```python
def softmax(z):
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-clasificacion-pipeline
fase: 04
leccion: 04
---

1. Custom <10K: ResNet-18 preentrenado, fine-tune FC.
2. Custom 10K-1M: ResNet-50 o EfficientNet-B3.
3. ImageNet: ConvNeXt, augmentation fuerte.
4. Few-shot: CLIP + linear probe.
5. Preentrenado siempre, AdamW + cosine.
```

## Ejercicios

1. **Mixup**: mezcla dos imagenes con alpha de Beta.
2. **CutMix**: reemplaza parche rectangular.
3. **Desafio**: entrena ResNet-18 en CIFAR-10 con
   augmentation y reporta top-1.

## Lecturas recomendadas

- "ImageNet Classification with Deep CNNs" (Krizhevsky et al., 2012)
- "torchvision" models: <https://pytorch.org/vision/stable/models.html>
- timm: <https://github.com/huggingface/pytorch-image-models>

---

> 📚 **Adaptación al español** de la lección "[Image Classification]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).