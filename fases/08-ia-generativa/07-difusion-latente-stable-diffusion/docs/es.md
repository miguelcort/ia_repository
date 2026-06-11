# Difusión latente y Stable Diffusion

> Stable Diffusion (Rombach 2022): VAE encode a latente 4×, U-Net diffusion en latente, CLIP text encoder, classifier-free guidance (CFG w=7.5). Variantes: SD 1.5 (512, 860M), SDXL (1024, 3.5B, dual encoder, refiner), SD3 (DiT, rectified flow), FLUX.1 (12B DiT, MM-DiT). Hoy: DiT + flow matching + controlnet + LoRA es SOTA en image y video.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/06-difusion-ddpm-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar VAE encode/decode.
- Implementar classifier-free guidance.
- Diagnosticar SD vs SDXL vs SD3.
- Aplicar text conditioning.

## Constrúyelo

```python
def classifier_free_guidance(eps_uncond, eps_cond, w):
    return eps_uncond + w * (eps_cond - eps_uncond)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-stable-diffusion
fase: 08
leccion: 07
---

1. VAE 8x, U-Net, CLIP, CFG.
2. CFG w=7.5 tipico.
3. SDXL: dual encoder, refiner.
4. SD3/FLUX: DiT, rectified flow.
5. Sora, SVD para video.
```

## Ejercicios

1. **CFG**: implementar dynamic CFG y
   comparar quality.
2. **SDXL-Lightning**: destilar SDXL a
   few-step.
3. **Desafio**: implementar text-to-video
   con SVD.

## Lecturas recomendadas

- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (Podell et al., 2023)
- "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Latent Diffusion Stable Diffusion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).