# 02 — Convoluciones desde cero

> La convolución es la operación que hace que las CNNs detecten patrones: bordes en capas tempranas, formas en intermedias, objetos en profundas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-imagen
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar convolución 2D desde cero con NumPy.
- Aplicar padding, stride y dilation.
- Diagnosticar campos receptivos y parámetros.
- Conectar convolución con multiplicación matricial.

## El problema

Una imagen RGB de 224x224 tiene 150,528 valores. Un MLP
fully-connected con 1000 neuronas en la primera capa
tendría 150M parámetros solo en esa capa. Una convolución
con kernel 3x3 tiene solo 27 parámetros. La convolución
aprovecha dos invarianzas: **localidad** (los píxeles
cercanos están correlacionados) y **traslación** (un borde
es un borde en cualquier posición). Esto es lo que hace
posible entrenar CNNs con millones de imágenes.

## El concepto

**Convolución 2D.** Para imagen `X` de tamaño `(H, W)` y
kernel `K` de tamaño `(k, k)`:

```text
Y[i, j] = Σ_{m, n} X[i + m, j + n] · K[m, n]
```

El kernel "se desliza" sobre la imagen y en cada posición
calcula una suma ponderada. En CNN, el kernel se aprende
vía backpropagation.

**Padding.** Sin padding, la salida es más pequeña que la
entrada (pierdes los píxeles del borde). Con `padding='same'`
añades ceros en el borde para mantener el tamaño. `padding=1`
añade 1 píxel de cada lado.

**Stride.** Por defecto el kernel se mueve 1 píxel a la vez.
Con `stride=2` se mueve 2 píxeles, lo que reduce la
dimensionalidad (downsampling). Las CNN modernas usan
convoluciones con stride en vez de pooling.

**Dilation (convolución atrous).** Inserta espacios entre
los elementos del kernel. `dilation=2` con kernel 3x3 ve
un campo receptivo efectivo de 5x5. Usado en segmentación
semántica (DeepLab) para expandir el campo receptivo sin
más parámetros.

**Canales múltiples.** Para imagen RGB (3 canales), el
kernel tiene forma `(k, k, C_in)`. Cada canal de salida se
calcula como una convolución 3D que suma sobre los `C_in`
canales. La capa convolucional con `C_out` filtros tiene
`k · k · C_in · C_out + C_out` parámetros.

**Campo receptivo.** La región de la imagen original que
influye en una activación. Una conv 3x3 tiene campo
receptivo 3x3; dos convs 3x3 tienen campo receptivo 5x5;
tres, 7x7. Las redes más profundas tienen campos
receptivos grandes sin tanto parámetro como las redes
anchas.

**Convolución como multiplicación matricial.** Una
convolución se puede expresar como `Y = W · X` donde `W` es
una matriz Toeplitz (muy sparse) construída a partir del
kernel. Es lo que hacen los frameworks internamente para
aprovechar BLAS.

**Implementación eficiente.**

- **im2col + GEMM:** convertir la imagen en columnas y
  multiplicar por el kernel extendido. Usado por Caffe.
- **Winograd:** reduce el número de multiplicaciones para
  kernels 3x3. Usado por NVIDIA.
- **FFT-based:** la convolución espacial = multiplicación
  en frecuencia. Útil para kernels grandes.

**Variantes.**

- **Convolución separable en profundidad (depthwise):** un
  kernel por canal. Reduce parámetros a `k · k · C_in`.
  Base de MobileNet.
- **Convolución separable puntual (pointwise):** conv 1x1
  sobre los canales. Reduce/meztura canales.
- **Convolución agrupada (grouped):** divide los canales
  en grupos y aplica convolución por grupo. Base de
  ResNeXt.
- **Convolución transpuesta (deconv):** "inversa" de la
  convolución. Usada en upsampling (segmentación, GAN).

**Cuándo usar cada variante.**

| Variante | Cuándo |
|---|---|
| Conv 2D estándar | Default en CNN |
| Depthwise separable | Mobile, edge (MobileNet, Xception) |
| Transpuesta | Upsampling (segmentación, GAN, autoencoder) |
| Dilated | Segmentación semántica (DeepLab) |
| Grouped | Computación eficiente (ResNeXt) |

## Constrúyelo

```python
import numpy as np


def conv2d(im, kernel, padding=0, stride=1):
    """Convolución 2D naive. im: (H, W). kernel: (kH, kW)."""
    H, W = im.shape
    kH, kW = kernel.shape
    if padding > 0:
        im = np.pad(im, padding, mode="constant", constant_values=0)
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = im[i * stride:i * stride + kH,
                         j * stride:j * stride + kW]
            out[i, j] = np.sum(region * kernel)
    return out


def conv2d_multichannel(im, kernel, padding=0, stride=1):
    """Convolución con canales. im: (H, W, C_in). kernel:
    (kH, kW, C_in). Salida: (out_H, out_W)."""
    C_in = im.shape[-1]
    assert kernel.shape[-1] == C_in
    H, W, _ = im.shape
    kH, kW, _ = kernel.shape
    if padding > 0:
        im = np.pad(im, ((padding, padding), (padding, padding), (0, 0)))
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = im[i * stride:i * stride + kH,
                         j * stride:j * stride + kW, :]
            out[i, j] = np.sum(region * kernel)
    return out


def conv_layer(im, kernels, padding=0, stride=1):
    """Capa convolucional con C_out filtros. im: (H, W, C_in).
    kernels: (kH, kW, C_in, C_out). Salida: (out_H, out_W, C_out)."""
    C_out = kernels.shape[-1]
    outputs = []
    for c in range(C_out):
        out = conv2d_multichannel(im, kernels[..., c], padding, stride)
        outputs.append(out)
    return np.stack(outputs, axis=-1)


def max_pool2d(im, size=2, stride=2):
    """Max pooling 2D. im: (H, W, C)."""
    H, W, C = im.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    out = np.zeros((out_H, out_W, C))
    for i in range(out_H):
        for j in range(out_W):
            region = im[i * stride:i * stride + size,
                         j * stride:j * stride + size, :]
            out[i, j, :] = region.max(axis=(0, 1))
    return out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-conv-arq
fase: 04
leccion: 02
---

Eres un asistente que ayuda a diseñar la arquitectura con-
volucional de una CNN. Recibirás el tamaño de input, la tarea,
y las restricciones de cómputo. Tu trabajo:

1. Si clasificación: ResNet-style, convs 3x3 con stride 2
   para downsampling.
2. Si mobile/edge: MobileNet con depthwise separable
   convs.
3. Si segmentación: U-Net con convs 2D, encoder-decoder
   con skip connections.
4. Si campo receptivo grande sin parámetros: dilated
   convs (DeepLab).
5. Si upsampling: convs transpuestas o bilinear + conv
   1x1.
6. Padding='same' para mantener dimensiones.
7. Stride=2 en vez de max pool para downsampling moderno.
8. Recomienda BatchNorm después de cada conv.
```

## Ejercicios

1. **Edge detection**: aplica Sobel y Laplacian a una
   imagen y visualiza los resultados.
2. **Max pool**: implementa max pooling 2x2 y observa el
   efecto de downsampling.
3. **Desafío**: implementa convolución separable en
   profundidad y compara parámetros con conv estándar.

## Lecturas recomendadas

- *Deep Learning* — Goodfellow et al. (cap. 9 sobre CNN).
- *CS231n* — Fei-Fei Li: <https://cs231n.github.io>.
- PyTorch nn.Conv2d: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>.

---

> 📚 **Adaptación al español** de la lección "[Convolutions From Scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
