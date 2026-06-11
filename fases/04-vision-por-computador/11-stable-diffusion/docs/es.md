# Stable Diffusion

> La primera difusion de calidad artistica que cabe en una GPU de consumo. Latent diffusion: la difusion opera en un espacio comprimido, no en pixeles. Es la base de la explosion de modelos generativos 2022-2024.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-generacion-de-imagenes-con-difusion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Conocer la arquitectura LDM (latent diffusion).
- Entender el flujo: text -> emb -> U-Net -> latent -> VAE -> imagen.
- Diagnosticar CFG y negative prompts.

## Constrúyelo

```python
def cfg(eps_cond, eps_uncond, w=7.5):
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
name: prompt-sd-elegir
fase: 04
leccion: 11
---

1. Rapido: SDXL-Turbo, LCM (4-8 pasos).
2. Max calidad: SD3, FLUX.1.
3. Custom: SDXL + LoRA.
4. CFG w=5-10, DPM++ 20-30 pasos.
5. diffusers + accelerate para self-host.
```

## Ejercicios

1. **LoRA desde cero**: implementa low-rank adaptation en
   una capa Linear.
2. **img2img**: implementa el flujo de difusion condicional
   a una imagen existente.
3. **Desafio**: fine-tunear SDXL con LoRA en 10 imagenes de
   un estilo artistico custom.

## Lecturas recomendadas

- "High-Resolution Image Synthesis with Latent Diffusion
  Models" (Rombach et al., 2022)
- "SDXL: Improving Latent Diffusion Models" (Podell et al., 2023)
- diffusers: <https://huggingface.co/docs/diffusers/>

---

> 📚 **Adaptación al español** de la lección "[Stable Diffusion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).