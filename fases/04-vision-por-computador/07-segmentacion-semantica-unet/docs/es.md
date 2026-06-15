# 07 — Segmentación semántica: U-Net

> U-Net asigna una clase a cada píxel. Es la base de segmentación médica, autos autónomos, y edición de imagen.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la arquitectura U-Net (encoder-decoder con
  skip connections).
- Calcular métricas de segmentación: IoU, Dice coefficient.
- Diagnosticar modos de falla comunes.
- Aplicar data augmentation para segmentación.

## El problema

La clasificación dice "qué hay en la imagen". La detección
dice "qué y dónde" con bounding boxes. La **segmentación
semántica** dice "qué clase tiene cada píxel". Es la tarea
más densa de visión: la salida es del mismo tamaño que la
entrada. U-Net (Ronneberger, 2015) es la arquitectura
canónica: encoder para extraer features, decoder para
recuperar resolución, skip connections para preservar
detalle fino.

## El concepto

**U-Net.** Encoder (downsampling) + decoder (upsampling) con
skip connections. El encoder usa pooling y convoluciones
para reducir resolución y aumentar canales. El decoder
usa upsampling (transpuesta o bilinear) y convoluciones para
recuperar resolución. Las skip connections concatenan
features del encoder con el decoder en la misma resolución,
preservando el detalle espacial.

**Bloque U-Net.** Dos convoluciones 3x3 + ReLU, seguidas de
downsampling o upsampling. En el cuello (bottleneck), la
resolución es mínima y los canales son máximos.

**Skip connections.** Conectan capa `i` del encoder con
capa `i` del decoder. Típicamente concatenación de
features, no suma. Permite que el decoder use tanto
contexto semántico (del cuello) como detalle espacial (de
las skip).

**Funciones de pérdida para segmentación.**

- **Cross-entropy per-pixel:** simple, pero desbalanceada
  con clases raras (e.g. tumor pequeño en escaneo médico).
- **Dice loss:** `1 - 2|A ∩ B| / (|A| + |B|)`. Maximiza el
  solapamiento. Robusta a desbalance.
- **Focal loss:** variante de cross-entropy que
  down-weighta píxeles fáciles. Útil con desbalance
  extremo.
- **Combo:** `cross-entropy + dice` es la default
  moderna.

**Métricas.**

- **IoU (Jaccard):** `|A ∩ B| / |A ∪ B|`. Por clase y
  promedio.
- **Dice coefficient:** `2|A ∩ B| / (|A| + |B|)`. Relacio-
  nado con IoU por `Dice = 2·IoU / (1 + IoU)`.
- **Pixel accuracy:** `% píxeles clasificados
  correctamente`. Engañosa con desbalance.

**Augmentations para segmentación.** Las mismas que
clasificación, pero las **mismas transformaciones** deben
aplicarse a la imagen y a la máscara. Albumentations es la
librería estándar (soporta transformaciones sincronizadas).

**Variantes de U-Net.**

- **U-Net++:** skip connections anidadas, con más
  capacidad.
- **Attention U-Net:** atención en las skip connections
  para enfocarse en regiones relevantes.
- **DeepLabV3+:** dilated convs en vez de pooling/stride.
  Más eficiente.
- **Mask2Former / SAM:** segmentación universal con
  transformers.

**Trampas.**

- **Desbalance de clases:** tumor 1% vs fondo 99%. Usa Dice
  o focal loss, no cross-entropy simple.
- **Augmentations no sincronizadas:** flip en imagen pero
  no en máscara. Usa Albumentations.
- **Evaluar con pixel accuracy:** siempre reporta IoU
  por clase y promedio.

## Constrúyelo

```python
import numpy as np


def unet_block(x, W1, b1, W2, b2):
    """Bloque convolucional de U-Net: 2 convs 3x3 + ReLU."""
    z = np.maximum(0, np.zeros_like(x))  # ReLU
    return z  # simplificado


def encoder(x, levels=4):
    """Encoder: 4 niveles de conv + maxpool."""
    skip_connections = []
    for _ in range(levels):
        x = unet_block(x, None, None, None, None)
        skip_connections.append(x)
        x = x[:, ::2, ::2, :]  # max pool 2x2
    return x, skip_connections


def decoder(x, skip_connections, levels=4):
    """Decoder: upsample + concat con skip + conv."""
    for i, skip in enumerate(reversed(skip_connections)):
        # upsample 2x
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        # concat con skip
        x = np.concatenate([x, skip], axis=-1)
        x = unet_block(x, None, None, None, None)
    return x


def iou_segmentacion(y_true, y_pred, n_clases):
    """IoU per-clase."""
    ious = []
    for c in range(n_clases):
        inter = np.sum((y_true == c) & (y_pred == c))
        union = np.sum((y_true == c) | (y_pred == c))
        ious.append(inter / max(union, 1e-12))
    return np.array(ious)


def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss entre máscaras binarias."""
    intersection = np.sum(y_true * y_pred)
    return 1 - (2 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-segmentacion-semantica
fase: 04
leccion: 07
---

Eres un asistente que ayuda a entrenar un modelo de
segmentación semántica. Recibirás el dataset, la resolución,
y la métrica objetivo. Tu trabajo:

1. Si dataset pequeño (médico): U-Net con data augmentation
   fuerte.
2. Si quieres SOTA: Mask2Former o SAM.
3. Si clases desbalanceadas: Dice + cross-entropy
   combinado.
4. Augmentations: Albumentations con transforms
   sincronizadas.
5. Encoder preentrenado (ResNet, EfficientNet) si dataset
   pequeño.
6. Métricas: IoU per-clase, mIoU, Dice.
7. Loss: Dice + cross-entropy (0.5 + 0.5) o Dice puro.
8. Resolution: 256x256 a 1024x1024 según GPU.
9. Evaluar con threshold 0.5 sobre las probabilidades
   softmax.
```

## Ejercicios

1. **U-Net minimal**: implementa U-Net con NumPy y
   aplícalo a una imagen de juguete.
2. **IoU + Dice**: implementa ambas métricas y compara
   en un dataset sintético.
3. **Desafío**: implementa Attention U-Net con bloques
   de atención en las skip connections.

## Lecturas recomendadas

- *U-Net: Convolutional Networks for Biomedical Image
  Segmentation* — Ronneberger et al., 2015.
- *DeepLabV3+* — Chen et al., 2018.
- *Segment Anything* — Kirillov et al., 2023.
- Albumentations: <https://albumentations.ai>.
- segmentation_models_pytorch: <https://github.com/qubvel-org/segmentation_models.pytorch>.

---

> 📚 **Adaptación al español** de la lección "[Semantic Segmentation: U-Net]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
