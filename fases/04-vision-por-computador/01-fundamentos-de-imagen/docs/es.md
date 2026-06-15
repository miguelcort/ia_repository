# 01 — Fundamentos de imagen

> Una imagen es una matriz de píxeles. Entender cómo se representa, almacena y transforma es prerequisito para todo en visión por computador.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-vectores-matrices-operaciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Representar imágenes como tensores (H, W, C).
- Convertir entre espacios de color (RGB, HSV, escala de
  grises).
- Aplicar filtros lineales y no lineales (blur, sharpen,
  bordes).
- Diagnosticar formatos de imagen (JPEG, PNG, TIFF) y
  muestreo.

## El problema

Una imagen en color es un tensor `(H, W, 3)` con valores
`uint8` (0-255) por canal. Antes de entrenar una CNN, hay
que normalizar a `float32 [0, 1]` o estandarizar. Antes de
visualizar, hay que desnormalizar y convertir a `uint8`.
La lección cubre el vocabulario mínimo para no perderse
en estos detalles.

## El concepto

**Tensor de imagen.** Forma `(H, W, C)` para una imagen
individual. En PyTorch espera `(C, H, W)`. `uint8` para
almacenar, `float32` para entrenar. Rango `[0, 255]` para
visualizar, `[0, 1]` o normalizado para entrenar.

**Espacios de color.**

- **RGB:** tres canales rojo, verde, azul. Default en
  cámaras y displays.
- **BGR:** igual que RGB pero con el orden invertido. OpenCV
  usa BGR por defecto.
- **HSV:** hue, saturation, value. Útil para segmentación por
  color.
- **Lab:** lightness, a, b. Perceptualmente uniforme; útil
  para matching de colores.
- **Escala de grises:** un canal, promedio ponderado de RGB
  (`0.299*R + 0.587*G + 0.114*B`).

**Filtros lineales.** Convolución 2D con un kernel:

- **Box filter (blur):** `[1/9, 1/9, 1/9; ...]`. Suaviza la
  imagen.
- **Gaussian blur:** kernel gaussiano. Más natural que box.
- **Sobel:** detector de bordes en X o Y.
- **Laplacian:** detector de bordes omnidireccional.
- **Sharpen:** `[0, -1, 0; -1, 5, -1; 0, -1, 0]`. Resalta
  bordes.

**Filtros no lineales.**

- **Median:** toma la mediana del vecindario. Bueno para
  eliminar ruido "salt and pepper".
- **Bilateral:** preserva bordes mientras suaviza. Más caro
  que Gaussian.
- **No-local means:** suaviza preservando estructuras.

**Formatos de archivo.**

- **JPEG:** con pérdida, no soporta transparencia. Bueno
  para fotos.
- **PNG:** sin pérdida, soporta transparencia. Bueno para
  gráficos.
- **TIFF:** sin pérdida, soporta capas. Estándar en
  publicación.
- **WebP:** moderno, mejor compresión que JPEG/PNG.

**Muestreo y cuantización.**

- **Muestreo espacial:** número de píxeles por unidad
  física (DPI/PPI).
- **Cuantización:** número de bits por píxel (8 bits = 256
  valores por canal).

**Histogramas.** Distribución de intensidades. Útil para
detectar imágenes mal expuestas y para *histogram
equalization* (mejora el contraste).

**Normalización para CNN.** Común:

- Dividir entre 255: rango `[0, 1]`.
- Estandarizar: `(x - mean) / std` con `mean=[0.485, 0.456,
  0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet).
- Ajustar a `[-1, 1]`: `(x - 0.5) / 0.5` (Tanh-ready).

## Constrúyelo

```python
import numpy as np


def rgb_a_grises(im):
    """Conversión RGB a escala de grises (ITU-R BT.601)."""
    return 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]


def rgb_a_hsv(im):
    """Conversión RGB a HSV."""
    im = im.astype(float) / 255
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    max_c = np.max(im, axis=-1)
    min_c = np.min(im, axis=-1)
    delta = max_c - min_c
    v = max_c
    s = np.where(max_c > 0, delta / np.maximum(max_c, 1e-12), 0)
    h = np.zeros_like(v)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / np.maximum(delta[mask_r], 1e-12)) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / np.maximum(delta[mask_g], 1e-12) + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / np.maximum(delta[mask_b], 1e-12) + 4
    h = h * 60  # a grados
    return np.stack([h, s, v], axis=-1)


def convolucion2d(im, kernel):
    """Convolución 2D naive. Para uso real, scipy o PyTorch."""
    h, w = im.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(im, pad, mode="reflect")
    out = np.zeros_like(im)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i + h, j:j + w] * kernel[i, j]
    return out


def sobel_x(im):
    return convolucion2d(im, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))


def gaussian_blur(im, ksize=5, sigma=1.0):
    k = ksize // 2
    x = np.arange(-k, k + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    kernel = np.outer(g, g)
    return convolucion2d(im, kernel)


def normalizar_imagen(im, mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)):
    """Normalización estándar de ImageNet."""
    im = im.astype(float) / 255.0
    return (im - np.array(mean)) / np.array(std)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-image-prep
fase: 04
leccion: 01
---

Eres un asistente que ayuda a preparar imágenes para una
CNN. Recibirás la imagen (path, formato, dimensiones) y
el modelo objetivo. Tu trabajo:

1. Cargar con Pillow o OpenCV.
2. Convertir a RGB (OpenCV usa BGR por defecto).
3. Redimensionar al tamaño esperado por el modelo
   (e.g. 224x224 para ResNet).
4. Normalizar según convención: ImageNet (mean, std)
   para modelos pre-entrenados.
5. Convertir a tensor `(C, H, W)` y batch.
6. Advertir contra normalizar con estadísticas del dataset
   si usas modelo pre-entrenado: usa siempre ImageNet.
```

## Ejercicios

1. **Conversión de color**: implementa `rgb_a_hsv` y
   `hsv_a_rgb` desde cero sin OpenCV.
2. **Filtros**: aplica Sobel y Gaussian blur a una imagen
   y visualiza los resultados.
3. **Desafío**: implementa histogram equalization en
   escala de grises.

## Lecturas recomendadas

- *Computer Vision: Algorithms and Applications* — Szeliski
  (cap. 2-3).
- *Digital Image Processing* — Gonzalez & Woods.
- OpenCV docs: <https://docs.opencv.org>.
- Pillow docs: <https://pillow.readthedocs.io>.

---

> 📚 **Adaptación al español** de la lección "[Image Fundamentals]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
