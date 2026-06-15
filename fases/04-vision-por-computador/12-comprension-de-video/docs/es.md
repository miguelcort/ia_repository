# 12 — Comprensión de video

> El video añade la dimensión temporal. Los modelos deben capturar movimiento, causalidad temporal, y consistencia frame a frame.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet,
                  14-vision-transformers
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar CNN 3D y video transformers.
- Aplicar temporal modeling (LSTM, attention) sobre
  features de cada frame.
- Diagnosticar consistencia temporal y flickering.
- Conocer datasets canónicos: Kinetics, Something-Something.

## El problema

Una imagen es un tensor `(H, W, C)`. Un video es una
secuencia `(T, H, W, C)`. La dimensión temporal añade
información (movimiento, causalidad) pero también
complejidad. Los modelos deben capturar relaciones entre
frames sin explotar la memoria. La lección cubre las
arquitecturas canónicas: CNN 3D, Two-Stream, I3D, SlowFast,
Video Swin, TimeSformer.

## El concepto

**CNN 3D.** Convolución con kernel `(kT, kH, kW)` que se
desliza en el tiempo y el espacio. Captura movimiento
local. Más parámetros que CNN 2D: una conv 3x3x3 con
256 canales de input y 256 de output tiene ~1.7M params
vs ~590k para 3x3 2D. Base de C3D, I3D.

**Two-Stream Networks (Simonyan & Zisserman, 2014).** Dos
CNNs: una sobre frames RGB, otra sobre optical flow. Las
predicciones se fusionan al final. Captura apariencia (RGB)
y movimiento (flow) por separado.

**I3D (Carreira & Zisserman, 2017).** Inflate una CNN 2D
preentrenada en ImageNet a 3D replicando los filtros a
través del tiempo. Preentrena en Kinetics. Es la base de
muchos modelos posteriores.

**SlowFast (Feichtenhofer et al., 2019).** Dos pathways:
Slow procesa frames a baja tasa temporal (e.g. 1 de cada
4) con muchos canales. Fast procesa todos los frames con
pocos canales. Las features de Fast se inyectan en Slow
vía conexiones laterales. Eficiente y accurado.

**Video Transformer / TimeSformer (Bertasius et al.,
2021).** Aplica atención dividida: spatial attention
(dentro de cada frame) y temporal attention (entre
frames) en bloques separados. Más escalable que
convoluciones 3D.

**Video Swin Transformer (Liu et al., 2022).** Extiende
Swin Transformer con atención temporal local
(ventanas temporales). Eficiente para videos largos.

**VideoMAE (Tong et al., 2022).** Masked autoencoding
para video: enmascara el 90% de los patches espacio-
temporales y reconstruye. Preentrenamiento
auto-supervisado SOTA.

**Datasets canónicos.**

- **Kinetics-400/600/700:** 400-700 clases de acciones
  humanas, 100k-500k clips.
- **Something-Something:** 174 acciones con objetos
  (empujar, tirar, etc.), enfoca en razonamiento
  temporal.
- **UCF-101:** 101 acciones, más pequeño.
- **HMDB-51:** 51 acciones.

**Métricas.** Top-1 y top-5 accuracy en clip-level
(promediado sobre los clips del test set), y
video-level (promediado sobre los T clips del video).

**Trampas.**

- **Memoria OOM:** un clip de 32 frames a 224x224 con
  batch 32 son ~150GB de memoria de GPU. Reducir batch,
  resolución o usar gradient checkpointing.
- **Flickering entre frames:** el modelo predice cada
  frame independientemente. Usar consistencia temporal
  (LSTM, GRU) o smoothing en post-procesado.
- **Training cost:** un video es 30-100x más caro que
  una imagen. Usar transfer learning desde
  preentrenados en Kinetics.

## Constrúyelo

```python
import numpy as np


def conv3d(x, kernel, padding=0, stride=1):
    """Convolución 3D naive. x: (T, H, W, C_in).
    kernel: (kT, kH, kW, C_in, C_out)."""
    T, H, W, C_in = x.shape
    kT, kH, kW, _, C_out = kernel.shape
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding),
                       (padding, padding), (0, 0)))
    out_T = (T + 2 * padding - kT) // stride + 1
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    out = np.zeros((out_T, out_H, out_W, C_out))
    for t in range(out_T):
        for i in range(out_H):
            for j in range(out_W):
                region = x[t * stride:t * stride + kT,
                             i * stride:i * stride + kH,
                             j * stride:j * stride + kW, :]
                out[t, i, j] = (region[..., None] * kernel).sum(
                    axis=(0, 1, 2, 3)
                )
    return out


def temporal_average_pool(features):
    """Promedia features a lo largo del tiempo.
    features: (T, D) -> (D,)."""
    return features.mean(axis=0)


def video_to_clip_preds(clip_probs):
    """Promedia probabilidades sobre los clips de un video.
    clip_probs: (T, n_clases)."""
    return clip_probs.mean(axis=0)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-video
fase: 04
leccion: 12
---

Eres un asistente que ayuda a entrenar un modelo de
clasificación de video. Recibirás el dataset, la duración
de los videos, y el hardware. Tu trabajo:

1. Si quieres accuracy SOTA: Video Swin o VideoMAE
   preentrenado en Kinetics.
2. Si quieres velocidad: SlowFast con 8 frames por clip.
3. Si dataset pequeño: I3D preentrenado, fine-tune.
4. Si GPU limitada: usar menos frames (8 vs 32) y
   menor resolución.
5. Augmentations: RandomCrop, HorizontalFlip,
   ColorJitter, RandomTemporalSubsample.
6. Preentrenar en Kinetics, fine-tune en tu dataset.
7. Evaluar en clip-level y video-level.
8. Si flickering: post-procesar con smoothing
   temporal o usar GRU entre frames.
```

## Ejercicios

1. **CNN 3D**: implementa una conv 3D y visualiza los
   filtros aprendiendo sobre Kinetics-400 mini.
2. **SlowFast**: implementa el pathway Slow + Fast y la
   fusión lateral.
3. **Desafío**: fine-tunea un VideoMAE preentrenado en
   un dataset pequeño.

## Lecturas recomendadas

- *SlowFast Networks for Video Recognition* —
  Feichtenhofer et al., 2019.
- *Is Space-Time Attention All You Need for Video
  Understanding?* — Bertasius et al., 2021.
- *Video Swin Transformer* — Liu et al., 2022.
- *VideoMAE* — Tong et al., 2022.
- PySlowFast: <https://github.com/facebookresearch/SlowFast>.

---

> 📚 **Adaptación al español** de la lección "[Video Understanding]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
