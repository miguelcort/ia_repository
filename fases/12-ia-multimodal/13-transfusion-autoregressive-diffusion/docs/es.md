# Transfusion autoregressive diffusion

> TransFusion (Zhou 2024): predict next token and diffuse images con one multi-modal model - text autoregresivo (cross-entropy loss sobre logits vs target ids) + image diffusion (DDPM schedule, predict noise, MSE loss predicted vs true noise) + shared transformer backbone. +Unified, -Dual architecture, +SOTA 2024-25. +Variants: 1B, 3B, 7B. Frameworks: original, transformers (HF), vLLM, diffusers. Image diffusion: forward (x_0 -> x_t via alpha_t*x_0 + sigma_t*noise), predict noise, loss MSE(predicted, true), reverse DDPM 1000 steps / DDIM 50 steps / flow matching 1 step (Stable Diffusion 3). TransFusion vs Emu3: TransFusion (AR + diffusion + quality -compute 1000 steps + continuous), Emu3 (AR only + simple + VQ-VAE 32768 + discrete). Decision: quality -> TransFusion, simple -> Emu3, any-to-any -> MIO, discrete diffusion -> Show-o. Production: MIO + TransFusion + Show-o SOTA 2024-25. Variants: TransFusion 1B-7B, Emu3 8B-12B, MIO, Show-o, Chameleon. Frameworks: original, transformers, vLLM, SGLang, diffusers. 2025: +Native + any-to-any + reasoning + video.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/11, 12/12
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar add_noise y diffusion_forward.
- Implementar diffusion_denoise_step.
- Implementar transfusion_loss_text y transfusion_loss_image.
- Implementar transfusion_step combined.
- Diagnosticar AR vs diffusion.

## Constrúyelo

```python
def add_noise(image, t, noise_schedule):
    rng = np.random.default_rng(t + 1)
    return image + noise_schedule[t] * rng.standard_normal(image.shape)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: transfusion
fase: 12
leccion: 13
---

1. AR text + image diffusion.
2. Shared transformer.
3. DDPM 1000 o DDIM 50.
4. Text CE + image MSE.
5. +Unified +SOTA.
```

## Ejercicios

1. **TransFusion**: usar
   TransFusion con HuggingFace.
2. **Diffusion**: entrenar
   diffusion en custom data.
3. **Desafio**: TransFusion
   para video generation.

## Lecturas recomendadas

- "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Zhou et al., 2024)
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Denoising Diffusion Implicit Models" (Song et al., 2020)
- "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser et al., 2024)

---

> 📚 **Adaptación al español de la lección [TransFusion Autoregressive Diffusion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).