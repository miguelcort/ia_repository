# 13 — Visión 3D: NeRFs

> NeRF (Neural Radiance Fields) representa una escena 3D como una red neuronal que mapea coordenadas 5D (x, y, z, theta, phi) a color y densidad.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Implementar NeRF desde cero con PyTorch.
- Aplicar hierarchical sampling y positional encoding.
- Renderizar vistas nuevas de una escena entrenada.
- Diagnosticar artefactos comunes.

## El problema

Para representar una escena 3D, las opciones tradicionales
son mallas poligonales o nubes de puntos. NeRF (Mildenhall
et al., 2020) usa una red neuronal que toma coordenadas
5D (3D + dirección de vista) y emite color (RGB) y
densidad (sigma). El modelo se entrena con muchas fotos
desde distintos ángulos. Una vez entrenado, puede
renderizar vistas nuevas de la escena. La calidad es
sorprendente: detalles finos, reflejos, transparencias.

## El concepto

**El modelo.** Un MLP `F_θ: (x, y, z, theta, phi) -> (RGB,
sigma)`. Input: posición 3D (x, y, z) y dirección de vista
(theta, phi) en coordenadas esféricas. Output: color RGB
y densidad volumétrica sigma.

**Renderizado por ray casting.** Para cada píxel, lanzar un
rayo desde la cámara. Muestrear N puntos a lo largo del
rayo. Para cada punto, consultar la red para obtener (RGB,
sigma). Acumular con la fórmula de renderizado volumétrico:

```text
C(r) = Σ T_i · (1 - exp(-σ_i · δ_i)) · c_i
T_i = exp(-Σ_{j<i} σ_j · δ_j)
```

donde `δ_i` es la distancia entre muestras consecutivas.

**Positional encoding.** Las coordenadas (x, y, z) tienen
poca capacidad expresiva en un MLP crudo. Aplicar
encoding sinusoidal:

```text
γ(p) = (sin(2^0 π p), cos(2^0 π p), sin(2^1 π p), cos(2^1 π p), ...)
```

Con 10 bandas para posición y 4 para dirección. Esto
permite al MLP aprender features de alta frecuencia.

**Hierarchical sampling.** Muestreo coarse-to-fine:
primero muestrear 64 puntos uniformes, identificar
regiones de alta densidad, y muestrear 128 puntos más
en esas regiones. Reduce cómputo y mejora calidad.

**Loss.** MSE entre color renderizado y color real:

```text
L = Σ_r ||C_pred(r) - C_gt(r)||²
```

**Tiempo de entrenamiento.** Original: ~1 día por escena
en una V100. Con aceleraciones (Instant-NGP, mip-NeRF 360,
Zip-NeRF): minutos a horas.

**Limitaciones.**

- **Por escena:** cada NeRF es un modelo separado. No
  generaliza entre escenas.
- **Tiempo de training:** lento. Aceleraciones recientes
  lo reducen.
- **Estática:** no modela escenas dinámicas. Para eso,
  Dynamic NeRF o 4D NeRF.
- **Memoria:** los campos de densidad son densos.

**Variantes y aceleraciones.**

- **Instant-NGP (Müller et al., 2022):** usa hash grids
  para representar la escena. Entrena en minutos.
- **mip-NeRF 360 (Barron et al., 2022):** maneja escenas
  no acotadas (outdoor).
- **Zip-NeRF:** combina grid + tri-plane.
- **3D Gaussian Splatting (Kerbl et al., 2023):**
  alternativa sin MLP, usa gaussianas explícitas.

**Trampas.**

- **Sampling muy denso:** costo computacional prohibitivo.
  Usar hierarchical sampling.
- **Sin positional encoding:** el MLP colapsa a una
  función suave, pierde detalles.
- **Vistas muy limitadas en training:** artefactos en
  regiones no vistas.

## Constrúyelo

```python
import numpy as np


def positional_encoding(x, L=10):
    """Encoding sinusoidal: senos y cosenos a múltiples
    frecuencias. x: (..., D), L: número de bandas."""
    freqs = 2.0 ** np.arange(L) * np.pi
    xb = x[..., None] * freqs  # (..., D, L)
    return np.concatenate([np.sin(xb), np.cos(xb)], axis=-1).reshape(*x.shape[:-1], -1)


def volumetric_rendering(sigma, color, deltas):
    """Renderiza un rayo. sigma: (N,), color: (N, 3),
    deltas: (N,). Devuelve RGB (3,)."""
    T = np.exp(-np.cumsum(sigma[:-1] * deltas[:-1]))
    T = np.concatenate([np.ones(1), T])  # T[0] = 1
    alpha = 1 - np.exp(-sigma * deltas)
    weights = T * alpha
    return (weights[..., None] * color).sum(axis=0)


def sample_along_ray(ro, rd, near=0.0, far=1.0, n_samples=64):
    """Muestrea n_points a lo largo del rayo. ro: (3,) origen,
    rd: (3,) dirección."""
    t = np.linspace(near, far, n_samples)
    return ro[None, :] + t[:, None] * rd[None, :]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nerf
fase: 04
leccion: 13
---

Eres un asistente que ayuda a entrenar un NeRF. Recibirás
las imágenes de la escena, las poses de cámara, y el
hardware. Tu trabajo:

1. Si quieres velocidad: Instant-NGP con hash grids.
2. Si quieres SOTA calidad: Zip-NeRF o 3D Gaussian
   Splatting.
3. Si la escena es outdoor no acotada: mip-NeRF 360.
4. Positional encoding con 10 bandas para posición, 4
   para dirección.
5. Hierarchical sampling: 64 coarse + 128 fine.
6. Loss: MSE sobre color, más opcional regularizer
   de sparsity.
7. Training: 100k-500k iterations.
8. Renderizar con 128-256 samples por rayo.
9. Evaluar con PSNR, SSIM, LPIPS sobre test set.
```

## Ejercicios

1. **Positional encoding**: implementa y visualiza cómo
   el encoding sinusoidal ayuda al MLP.
2. **Volumetric rendering**: implementa el renderizado
   y verifica que da la imagen original para una escena
   simple.
3. **Desafío**: entrena un NeRF sobre las imágenes de
   la LEGO NeRF dataset.

## Lecturas recomendadas

- *NeRF: Representing Scenes as Neural Radiance Fields for
  View Synthesis* — Mildenhall et al., 2020.
- *Instant Neural Graphics Primitives* — Müller et al.,
  2022.
- *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering* — Kerbl et al., 2023.
- nerfstudio: <https://docs.nerf.studio>.

---

> 📚 **Adaptación al español** de la lección "[3D Vision: NeRFs]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
