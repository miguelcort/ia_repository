# 03 — CNNs: de LeNet a ResNet

> Las CNNs evolucionaron en 25 años: LeNet (1998) -> AlexNet (2012) -> VGG (2014) -> ResNet (2015). ResNet sigue siendo la columna vertebral de la visión moderna.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-convoluciones-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar las arquitecturas canónicas: LeNet, AlexNet,
  VGG, ResNet.
- Explicar por qué ResNet (skip connections) permite redes
  más profundas.
- Diagnosticar vanishing gradient y cómo las skip connections
  lo resuelven.
- Conocer las variantes modernas: ResNeXt, EfficientNet,
  ConvNeXt.

## El problema

¿Cómo diseñar una CNN que sea profunda sin que diverja o
sobreajuste? La historia de la visión por computador es
una saga de innovaciones arquitectónicas: LeNet demostró
que las CNNs funcionan para dígitos; AlexNet ganó
ImageNet 2012 con ReLU y dropout; VGG mostró que más
profundidad (16-19 capas) ayuda; ResNet llevó la
profundidad a 152+ capas con skip connections. La lección
cubre las cuatro y por qué cada innovación fue necesaria.

## El concepto

**LeNet-5 (LeCun et al., 1998).** Primera CNN exitosa
aplicada a dígitos MNIST. Arquitectura: conv 5x5 -> pool
-> conv 5x5 -> pool -> FC -> FC. Demostró que las CNNs
superan a otros métodos en tareas de visión.

**AlexNet (Krizhevsky et al., 2012).** Ganó ImageNet 2012
por un margen enorme. Innovaciones: ReLU (en vez de
tanh), dropout, data augmentation, GPU training. 60M
parámetros. Marcó el inicio del deep learning moderno.

**VGG (Simonyan & Zisserman, 2014).** Apiló muchas
convoluciones 3x3 en secuencia. VGG-16 y VGG-19: 16 y 19
capas con pesos. Demostró que la profundidad ayuda. Pero
los gradientes se desvanecen en redes tan profundas sin
skip connections.

**ResNet (He et al., 2015).** Resolvió el problema de
degradación (más capas = peor accuracy en train, no
solo en val) con **skip connections**: la salida de un
bloque es `F(x) + x` en vez de `F(x)`. Esto permite que
la red aprenda **residuos** pequeños sobre la identidad.

**El bloque residual.** Para entrada `x`:

```text
y = F(x, {W_i}) + x
```

donde `F = W_2 · ReLU(W_1 · x + b_1) + b_2`. Si la
transformación óptima es cercana a la identidad, la red
puede aprender `F ≈ 0` fácilmente. Esto facilita el
entrenamiento de redes profundas.

**¿Por qué funcionan las skip connections?**

1. **Gradientes:** durante backprop, el gradiente fluye por
   la skip connection sin atenuarse, evitando vanishing.
2. **Optimización:** aprender `F(x) = 0` es más fácil que
   aprender `F(x) = x` (la identidad).
3. **Ensemble implícito:** ResNet puede verse como un
   ensemble de muchas sub-redes de diferente profundidad
   (Veit et al., 2016).

**Variantes.**

- **ResNeXt:** introduce "cardinality" como tercera
  dimensión (junto a depth y width). Convs grouped.
- **Wide ResNet (WRN):** más canales, menos profundidad.
- **DenseNet:** cada capa se conecta con todas las
  siguientes. Más denso que ResNet.
- **EfficientNet:** scaling compuesto (depth, width,
  resolution simultáneamente).
- **ConvNeXt (2022):** "modernizar" ResNet con
  ideas de transformers (depthwise convs grandes, GELU,
  LayerNorm). Supera a Swin Transformer en muchos
  benchmarks.

**Patrones arquitectónicos modernos.**

- **Stem:** primeras capas reducen la resolución (conv
  7x7 stride 2).
- **Stages:** 4 stages de ResNet blocks, con downsampling
  entre stages (stride 2).
- **Head:** average pool + linear para clasificación.
- **Residual connections:** en cada bloque.
- **BatchNorm + ReLU (o GELU):** después de cada conv.

**Trampas.**

- **Más profundidad sin motivo:** más allá de ~50 capas,
  gains marginales. Mejor aumentar width o resolución.
- **Skip connections mal implementadas:** el shape de `F(x)`
  y `x` debe coincidir. Si downsampling cambia el tamaño,
  añade una conv 1x1 en la skip.
- **BatchNorm antes/después de la skip:** el orden importa.
  Post-activation (conv -> BN -> ReLU -> add) es el más
  estable.

## Constrúyelo

```python
import numpy as np


def relu(x):
    return np.maximum(0, x)


def conv2d_simple(x, kernel, padding=0):
    """Convolución 2D simplificada sin stride."""
    H, W = x.shape
    kH, kW = kernel.shape
    if padding > 0:
        x = np.pad(x, padding, mode="constant")
    out = np.zeros((H + 2 * padding - kH + 1, W + 2 * padding - kW + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(
                x[i:i + kH, j:j + kW] * kernel
            )
    return out


def resnet_block(x, W1, b1, W2, b2):
    """Bloque residual: y = ReLU(W2·ReLU(W1·x+b1)+b2) + x, luego ReLU."""
    z1 = relu(conv2d_simple(x, W1) + b1)
    z2 = conv2d_simple(z1, W2) + b2
    if z2.shape != x.shape:
        # Skip con proyección 1x1 (downsampling)
        W_skip = np.zeros((1, 1))
        x_proj = conv2d_simple(x, W_skip)
        if x_proj.shape != z2.shape:
            x_proj = x_proj[:z2.shape[0], :z2.shape[1]]
    else:
        x_proj = x
    return relu(z2 + x_proj)


def simple_resnet_forward(x, layers, W_out, b_out):
    """Forward simplificado de una ResNet."""
    for layer in layers:
        x = layer(x)
    # average pool + linear
    x = x.mean(axis=(0, 1))
    return x @ W_out + b_out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-cnn-arq
fase: 04
leccion: 03
---

Eres un asistente que ayuda a elegir arquitectura CNN. Reci-
birás la tarea (clasificación/segmentación), tamaño del
dataset, y restricciones. Tu trabajo:

1. Si dataset mediano (100k imágenes) y quieres lo
   último: ConvNeXt o EfficientNet preentrenado.
2. Si mobile/edge: MobileNetV3 o EfficientNet-B0.
3. Si quieres entrenar desde cero en ImageNet: ResNet-50
   o ResNet-152 con skip connections.
4. Si segmentación: U-Net o DeepLabV3+.
5. Si detección: backbone ResNet + FPN + detector
   (RetinaNet o YOLO).
6. Recomienda siempre transfer learning en lugar de
   entrenar desde cero.
7. Advertir contra arquitecturas > 200 capas: gains
   marginales, costo computacional alto.
```

## Ejercicios

1. **Bloque residual**: implementa el bloque residual
   básico y compara gradientes con/sin skip connection.
2. **ResNet-18 mínimo**: implementa ResNet-18 en
   PyTorch y entrena en CIFAR-10.
3. **Desafío**: implementa ConvNeXt-T desde cero y
   compara accuracy vs parámetros vs ResNet-50.

## Lecturas recomendadas

- *Deep Residual Learning* — He et al., 2015.
- *VGG Very Deep Convolutional Networks* — Simonyan &
  Zisserman, 2014.
- *ImageNet Classification with Deep CNNs* — Krizhevsky
  et al., 2012.
- *A ConvNet for the 2020s* — Liu et al., 2022 (ConvNeXt).
- torchvision models: <https://pytorch.org/vision/stable/models.html>.

---

> 📚 **Adaptación al español** de la lección "[CNNs: From LeNet to ResNet]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
