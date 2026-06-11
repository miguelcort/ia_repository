# Visión 3D con NeRF

> Neural Radiance Fields: una red aprende a representar una escena 3D como una funcion (color, densidad) sobre cada punto del espacio. Volumen rendering convierte esa funcion en imagenes vistas desde cualquier angulo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-clasificacion-de-imagenes
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar proyeccion pinhole y rayos.
- Muestrear puntos a lo largo de un rayo.
- Implementar composicion de volumen (alpha blending).

## Constrúyelo

```python
def composicion_volumen(densidad, color, t, delta):
    alpha = 1.0 - np.exp(-densidad * delta)
    T = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    pesos = T * alpha
    return (pesos[..., None] * color).sum(axis=-2)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-3d-recon
fase: 04
leccion: 13
---

1. Estatica: 3D Gaussian Splatting.
2. Dinamica: 4DGS, NeRFies.
3. Poses: COLMAP.
4. Rapido: Instant-NGP hash encoding.
5. nerfstudio framework.
```

## Ejercicios

1. **Positional encoding**: anade senos/cosenos de
   multiples frecuencias.
2. **Hash encoding**: implementa multi-resolution hash
   (Instant-NGP).
3. **Desafio**: implementar 3D Gaussian Splatting basico
   y entrenar en una escena pequena.

## Lecturas recomendadas

- "NeRF" (Mildenhall et al., 2020)
- "Instant-NGP" (Mueller et al., 2022)
- "3D Gaussian Splatting" (Kerbl et al., 2023)
- nerfstudio: <https://docs.nerf.studio/>

---

> 📚 **Adaptación al español** de la lección "[3D Vision with NeRF]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).