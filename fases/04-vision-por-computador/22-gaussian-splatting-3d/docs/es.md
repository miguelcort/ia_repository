# Gaussian Splatting 3D

> 100x mas rapido que NeRF con calidad similar: 30+ FPS a 1080p. La escena es una nube de ~1M de gaussianas 3D optimizadas con rasterizacion diferenciable. Estado del arte para reconstruccion de escenas estaticas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13-vision-3d-nerf
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Inicializar gaussianas 3D (pos, escala, color, opacidad).
- Proyectar a 2D con matriz de camara.
- Hacer alpha compositing.
- Diagnosticar spherical harmonics y covarianza.

## Constrúyelo

```python
def inicializar_gaussiana(pos, escala=0.1, color=(1.0, 0.0, 0.0), opacidad=1.0):
    return {
        "pos": np.array(pos, dtype=np.float32),
        "escala": np.array([escala, escala, escala], dtype=np.float32),
        "color": np.array(color, dtype=np.float32),
        "opacidad": float(opacidad),
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-3dgs-elegir
fase: 04
leccion: 22
---

1. Estatica: 3DGS (rapido, alta calidad).
2. Dinamica: 4DGS.
3. Geometria precisa: NeRF + marching cubes.
4. VR/AR: 3DGS viewer (gsplat.js).
5. Poses: COLMAP, 50-100 imagenes.
```

## Ejercicios

1. **Covarianza 3D**: implementar proyeccion analitica de
   covarianza 3D a elipsoide 2D.
2. **Densificacion**: anadir gaussianas en regiones con
   gradiente alto.
3. **Desafio**: implementar 3DGS basico desde cero y
   entrenar en una escena pequena (e.g. lego).

## Lecturas recomendadas

- "3D Gaussian Splatting for Real-Time Radiance Field
  Rendering" (Kerbl et al., 2023)
- gsplat: <https://docs.gsplat.studio/main/>
- nerfstudio: <https://docs.nerf.studio/>

---

> 📚 **Adaptación al español** de la lección "[3D Gaussian Splatting]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).