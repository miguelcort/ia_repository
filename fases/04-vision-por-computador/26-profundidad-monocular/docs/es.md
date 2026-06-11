# Profundidad monocular

> Dada una sola imagen RGB, predecir la distancia a cada pixel. Depth Anything V2 (relativo) y Metric3D V2 (métrico) marcan el estado del arte. Aplicaciones: AR, robotics, Bokeh, autonomous driving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-clasificacion-de-imagenes
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Convertir depth + intrinsics a puntos 3D.
- Implementar métricas (delta<1.25, AbsRel, RMSE, MAE).
- Diagnosticar relativo vs métrico.

## Constrúyelo

```python
def depth_to_3d(depth, K, xy):
    u, v = xy
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth[int(v), int(u)]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-depth-elegir
fase: 04
leccion: 26
---

1. Max calidad: Depth Anything V2.
2. Metrico indoor: ZoeDepth.
3. Metrico outdoor: Metric3D V2.
4. Mobile: MiDaS small.
5. Custom video: Monodepth2.
```

## Ejercicios

1. **Point cloud**: depth + intrinsics -> nube de puntos 3D.
2. **3D photo effect**: parallax via depth + warping.
3. **Desafio**: usar Depth Anything V2 en un video
   propio y generar point cloud 3D.

## Lecturas recomendadas

- "Depth Anything V2" (Yang et al., 2024)
- "MiDaS" (Ranftl et al., 2020)
- "Monodepth2" (Godard et al., 2019)

---

> 📚 **Adaptación al español** de la lección "[Monocular Depth]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).