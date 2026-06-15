# 22 — 3D Gaussian Splatting

> 3DGS (Kerbl et al., 2023) representa escenas 3D con millones de gaussianas 3D optimizables. Reemplaza a NeRF con calidad similar y entrenamiento 100x más rápido.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13-vision-3d-nerf
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entender la representación de gaussianas 3D
  optimizables.
- Implementar el differentiable rasterizer simplificado.
- Diagnosticar artefactos y número óptimo de gaussianas.
- Conocer variantes: 4D Gaussian Splatting, Mip-Splatting.

## El problema

NeRF entrena una red neuronal que representa la escena
implícitamente. Es lento (~1 día por escena) y la
representación es opaca. **3D Gaussian Splatting** (3DGS,
Kerbl et al., 2023) usa millones de gaussianas 3D
explícitas (cada una con posición, color, opacidad, escala,
rotación) y las optimiza con descenso de gradiente sobre
fotos. Entrena en minutos, renderiza en tiempo real (30+
FPS) y es editable.

## El concepto

**Representación.** Cada gaussiana 3D tiene:

- Posición `(x, y, z)`.
- Color RGB.
- Opacidad `α ∈ [0, 1]`.
- Escala `(s_x, s_y, s_z)`.
- Rotación (cuaternión unitario).

Total: ~14 floats por gaussiana. Una escena típica tiene
1-5M de gaussianas (~50-200 MB).

**Inicialización.** Desde una nube de puntos (SfM o
random), se inicializan gaussianas pequeñas. Luego se
optimiza la posición, color, opacidad, escala, rotación
de cada gaussiana vía descenso de gradiente.

**Rasterizer diferenciable.** Para cada píxel, se
proyectan las gaussianas 3D a 2D (proyección perspectiva
de elipsoides) y se hace alpha-blending en orden de
profundidad. La diferenciabilidad permite backprop a
través de la imagen renderizada.

**Densificación y pruning.** Cada N iteraciones:

- **Densificar:** clonar gaussianas grandes (subjetendidas)
  o partir gaussianas grandes (sobre-extendidas).
- **Podar:** eliminar gaussianas con opacidad < threshold.

Esto adapta la resolución de la representación a la
complejidad de la escena.

**Loss.** Combina L1 y SSIM-D entre imagen renderizada y
ground truth:

```text
L = (1 - λ) · L1 + λ · L1_SSIM
```

**Ventajas vs NeRF.**

- Entrenamiento 100x más rápido (minutos vs días).
- Renderizado en tiempo real (30+ FPS).
- Editable: puedes mover gaussianas, cambiar colores.
- Memoria explícita: sabes qué representa cada gaussiana.

**Limitaciones.**

- Memoria: 1-5M gaussianas son 50-200 MB.
- Estática: para escenas dinámicas, usar 4D-GS.
- Bordes: a veces se ven "huecos" en regiones no vistas.

**Cuándo usar NeRF vs 3DGS.**

| Caso | Recomendación |
|---|---|
| Tiempo real, edición | 3DGS |
| Calidad SOTA, una escena | NeRF + Zip-NeRF |
| Memoria limitada | NeRF (red) > 3DGS (explícita) |
| Edición creativa | 3DGS |

**Trampas.**

- **Inicialización pobre:** sin SfM, la calidad baja.
  Usar COLMAP para generar la nube de puntos inicial.
- **Densificación mal calibrada:** muchas o pocas
  gaussianas. Ajustar `densify_from_iter` y
  `densify_until_iter`.
- **Tiempo de entrenamiento:** 30k iteraciones son
  típicas, ~30 min en una A100. Más iteraciones
  mejoran marginal.

## Constrúyelo

```python
import numpy as np


class Gaussian:
    """Gaussiana 3D simplificada."""
    def __init__(self, position, color, opacity=1.0,
                 scale=(1.0, 1.0, 1.0)):
        self.position = np.array(position)
        self.color = np.array(color)
        self.opacity = opacity
        self.scale = np.array(scale)


def project_gaussians(gaussians, K, pose):
    """Proyecta gaussianas 3D a 2D usando la matriz intrínseca K
    y la pose de cámara. Simplificado: solo posición."""
    projected = []
    for g in gaussians:
        # Transformar al espacio de cámara
        p_cam = pose[:3, :3] @ g.position + pose[:3, 3]
        # Proyectar a imagen
        if p_cam[2] > 0:
            p_img = K @ (p_cam / p_cam[2])
            projected.append((p_img[:2], g.color, g.opacity))
    return projected


def alpha_blend(projected_gaussians, H, W):
    """Alpha-blending de gaussianas en orden de profundidad."""
    img = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    for pos, color, op in sorted(
        projected_gaussians, key=lambda x: -x[2]
    ):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = color * op + img[y, x] * (1 - op)
            alpha[y, x] = op + alpha[y, x] * (1 - op)
    return img
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-3dgs
fase: 04
leccion: 22
---

Eres un asistente que ayuda a usar 3D Gaussian Splatting.
Recibirás los videos de la escena, las poses de cámara
(calculadas con COLMAP), y el hardware. Tu trabajo:

1. Inicialización con COLMAP para nube de puntos
   sparse.
2. Entrenar 30k iteraciones con 3DGS.
3. Densificación cada 500 iteraciones hasta 15k.
4. Pruning de gaussianas con opacidad < 0.005.
5. Loss: 0.8 L1 + 0.2 SSIM-D.
6. Evaluar con PSNR, SSIM, LPIPS.
7. Si la escena es dinámica: 4D-GS.
8. Para edición: manipular las gaussianas directamente
   (mover, cambiar color).
9. Si memoria es limitante: comprimir gaussianas o
   usar 2D-GS.
```

## Ejercicios

1. **Inicialización**: implementa la inicialización desde
   una nube de puntos.
2. **Alpha blending**: implementa el differentiable
   rasterizer simplificado.
3. **Desafío**: entrena 3DGS en un dataset de NeRF
   Studio y mide PSNR vs NeRF.

## Lecturas recomendadas

- *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering* — Kerbl et al., 2023.
- *4D Gaussian Splatting* — Wu et al., 2024.
- *Mip-Splatting* — Yu et al., 2024.
- gsplat: <https://github.com/nerfstudio-project/gsplat>.

---

> 📚 **Adaptación al español** de la lección "[3D Gaussian Splatting]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
