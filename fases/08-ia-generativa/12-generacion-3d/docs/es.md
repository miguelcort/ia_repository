# Generación 3D

> 3D generation: NeRF (MLP density+color, volumetric), Gaussian Splatting (3D Gaussians explícitas, real-time 30+ fps), DreamFusion (text-to-3D via SDS), LRM (feed-forward 1s). Variantes: point clouds, mesh diffusion, 4D-GS. Aplicaciones: VFX, games, AR/VR, robotics, digital twins, architecture, medical, e-commerce. Hoy: feed-forward 3D + Gaussian Splatting es estándar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/09-vision-transformers
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar generación de rayos.
- Implementar volumetric rendering.
- Inicializar 3D Gaussians.
- Comparar NeRF vs GS vs DreamFusion vs LRM.

## Constrúyelo

```python
def volumetric_render(rgb, sigma, t_vals):
    delta = np.concatenate([t_vals[1:] - t_vals[:-1], [1e10]])
    alpha = 1.0 - np.exp(-sigma.squeeze() * delta)
    T = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    weights = alpha * T
    return (weights[..., None] * rgb).sum(axis=-2)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-3d-generation
fase: 08
leccion: 12
---

1. NeRF: MLP + volume render.
2. Gaussian Splatting: 3D gaussians.
3. DreamFusion: SDS.
4. LRM: feed-forward.
5. Apps: VFX, games, robotics.
```

## Ejercicios

1. **NeRF**: entrenar NeRF en custom
   scene con fotos.
2. **GS**: capturar escena con
   Gaussian Splatting.
3. **Desafio**: implementar text-to-3D
   con SDS.

## Lecturas recomendadas

- "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al., 2020)
- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
- "DreamFusion: Text-to-3D using 2D Diffusion" (Poole et al., 2022)
- "LRM: Large Reconstruction Model for Single Image to 3D" (Hong et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[3D Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).