# Comprensión de video

> Video = secuencia temporal de imagenes. Modelos: 3D CNN, (2+1)D, SlowFast, Video Transformer. Tarea clave: action recognition, detection, captioning, generation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-stable-diffusion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Cargar y muestrear frames de video.
- Calcular diferencia entre frames.
- Diagnosticar tareas de video understanding.

## Constrúyelo

```python
def muestrear_frames(video, n=5, estrategia="uniforme"):
    T = len(video)
    idx = np.linspace(0, T - 1, n, dtype=int)
    return video[idx], idx
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-video-tarea
fase: 04
leccion: 12
---

1. Action recognition: Video Swin, SlowFast.
2. Deteccion: tube + SORT/DeepSORT.
3. Captioning: BLIP-2, VideoLLaMA.
4. Generation: AnimateDiff, SVD, Sora.
5. Kinetics preentrenado.
```

## Ejercicios

1. **Optical flow real**: implementa Lucas-Kanade.
2. **Video classification**: 3D CNN sobre UCF-101 subset.
3. **Desafio**: fine-tunear SlowFast en un dataset
   custom de acciones.

## Lecturas recomendadas

- "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019)
- "Video Swin Transformer" (Liu et al., 2022)
- decord: <https://github.com/dmlc/decord>

---

> 📚 **Adaptación al español** de la lección "[Video Understanding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).