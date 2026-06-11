# Generación de video

> SOTA: Sora (DiT spacetime patches, 60s 1080p), Veo 2 (Google, 4K), Runway Gen-3, Kling, Pika. SVD (Stability, image-to-video, 14-25 frames). Arquitecturas: DiT spacetime, MM-DiT, 3D U-Net, AnimateDiff, CogVideoX. Métricas: FVD, IS, CLIP score, subject consistency, motion smoothness. Frontier: world models, real-time, interactive.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/07-difusion-latente-stable-diffusion
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar spacetime patches.
- Implementar causal temporal attention.
- Diagnosticar componentes de Sora.
- Comparar SVD, Sora, AnimateDiff.

## Constrúyelo

```python
def spacetime_patches(video, patch_size=2, temporal_patch=2):
    T, H, W, C = video.shape
    patches = []
    for t in range(0, T, temporal_patch):
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                p = video[t:t+temporal_patch, i:i+patch_size, j:j+patch_size, :]
                patches.append(p.flatten())
    return np.stack(patches)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-video-generation
fase: 08
leccion: 10
---

1. Sora: DiT spacetime, 3D VAE, RoPE 3D.
2. SVD: image-to-video, 14-25 frames.
3. Metrics: FVD, CLIP score.
4. AnimateDiff: motion LoRA.
5. Sora, Veo 2, Runway SOTA.
```

## Ejercicios

1. **SVD**: aplicar SVD a imagen
   custom y medir FID.
2. **AnimateDiff**: agregar motion LoRA
   a SD.
3. **Desafio**: implementar CogVideoX
   para text-to-video.

## Lecturas recomendadas

- "Video Generation Models as World Simulators" (Brooks et al., 2024)
- "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (Blattmann et al., 2023)
- "AnimateDiff: Training Your Text-to-Image Diffusion Models without Tuning" (Guo et al., 2023)
- "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (Yang et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Video Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).