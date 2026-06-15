# 07 — Difusión latente: Stable Diffusion

> Stable Diffusion (Rombach et al., 2022) combina un autoencoder variacional con un modelo de difusión en el espacio latente. Permite generar imágenes de alta calidad con poca memoria GPU, y se ha convertido en la base de la mayoría de modelos de texto-a-imagen open-source.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-difusion-ddpm-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el pipeline de Stable Diffusion: VAE encoder
  + U-Net difusión + VAE decoder.
- Implementar text conditioning con CLIP.
- Conocer las variantes: SD 1.5, SDXL, SD3, SDXL-Turbo.
- Diagnosticar cuándo usar Stable Diffusion y cuándo
  no.

## El problema

Los modelos de difusión pixel-space son lentos y
requieren mucha memoria. Stable Diffusion (Rombach et
al., 2022) hace difusión en el espacio latente de un
VAE pre-entrenado: la imagen se comprime a un latente
48×48 (vs 512×512), y la difusión opera en el latente.
Esto reduce cómputo 48x y permite generación en
consumer GPUs. La lección cubre el pipeline y las
variantes modernas.

## El concepto

**Pipeline de Stable Diffusion.**

1. **VAE encoder:** imagen 512×512×3 → latente
   64×64×4. Pre-entrenado en imágenes naturales.
2. **U-Net difusión:** opera en el espacio latente
   64×64×4. Condicionado en text embeddings.
3. **VAE decoder:** latente 64×64×4 → imagen 512×512×3.

**Text conditioning con CLIP.**

- **CLIP text encoder:** prompt → embeddings 77×768
  (SD 1.5) o 77×1280 (SDXL).
- **Cross-attention:** U-Net atiende a estos embeddings.
  Permite que el ruido predicho sea guiado por el
  texto.
- **Classifier-free guidance:** durante training, drop
  text conditioning 10% del tiempo. En inference,
  combinar predicción condicional y no-condicional:
  `eps = eps_uncond + scale * (eps_cond - eps_uncond)`.
  Scale=7.5 es típico.

**Variantes.**

- **SD 1.5 (Rombach et al., 2022):** 860M params, base
  para la mayoría de fine-tunes.
- **SD 2.x:** CLIP H, mayor resolución, mejor
  coherencia.
- **SDXL (Podell et al., 2023):** 2.6B params, dual
  text encoders, refiner para high-freq details.
- **SD3 (Esser et al., 2024):** DiT (diffusion
  transformer) en vez de U-Net. MM-DiT, mejor texto.
- **SDXL-Turbo (Sauer et al., 2023):** destilación
  adversarial, 1-4 steps.

**ControlNet (Zhang et al., 2023).** Añade conditioning
adicional al U-Net: edge maps, depth, pose, style. Sin
re-entrenar el modelo base. Permite control fino de
generación.

**LoRA y fine-tuning.**

- **LoRA (Hu et al., 2021):** low-rank adaptation. Solo
  entrenar 0.1-1% de parámetros para custom style/
  subject.
- **DreamBooth (Ruiz et al., 2023):** fine-tuning
  completo con un nuevo subject ("[V] token").
- **Textual inversion:** aprender un nuevo token
  embedding para un concept.

**Cuándo usar Stable Diffusion.**

- **Texto-a-imagen:** SDXL, SD3, DALL-E, Imagen.
- **Inpainting, outpainting, super-resolución.**
- **Style transfer:** con LoRA o textual inversion.
- **Customización:** LoRA, DreamBooth.

**Cuándo NO usar.**

- **Para texto:** usar LLMs. La difusión no modela
  lenguaje bien.
- **Para 3D:** usar NeRF + diffusion o 3D-aware GANs.
- **Para video:** usar video diffusion (Sora, Veo,
  AnimateDiff).
- **Para precisión:** modelos de difusión no son
  precisos pixel-by-pixel.

**Trampas.**

- **Sin CFG scale:** images no siguen el prompt. Usar
  scale=7-15.
- **Modelo no alineado al prompt:** CLIP text encoder
  no entiende fine details. Reformular el prompt.
- **Sampling steps muy bajos:** degrada calidad. T=20-50
  con DPM-Solver o T=1-4 con destilación.

## Constrúyelo

```python
import numpy as np


def encode_text(prompt, clip_model, max_length=77):
    """Codifica prompt a embeddings con CLIP text encoder.
    Devuelve (max_length, d_text)."""
    tokens = clip_model.tokenize(prompt, max_length)
    return clip_model.encode_text(tokens)


def classifier_free_guidance(eps_cond, eps_uncond, scale=7.5):
    """CFG: eps = eps_uncond + scale * (eps_cond - eps_uncond)."""
    return eps_uncond + scale * (eps_cond - eps_uncond)


def decode_latent(z, vae_decoder):
    """Decodifica latente a imagen con VAE decoder."""
    return vae_decoder(z)
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

Eres un asistente que ayuda con Stable Diffusion.
Recibirás la tarea. Tu trabajo:

1. Para texto-a-imagen: SDXL, SD3, o DALL-E 3.
2. Para inpainting: SD inpainting pipeline.
3. Para style transfer: LoRA + textual inversion.
4. Para custom subject: DreamBooth o LoRA.
5. Para control (edge, depth, pose): ControlNet.
6. Para velocidad: SDXL-Turbo, LCM.
7. Para evaluar: CLIP score, FID, human preference.
8. Advertir contra copyright, deepfakes, y bias.
```

## Ejercicios

1. **SD inference**: usa el pipeline de Hugging Face
   para generar una imagen.
2. **CFG scale**: visualiza cómo scale=1 vs scale=15
   afecta la calidad.
3. **Desafío**: entrena un LoRA en 10 imágenes de tu
   mascota.

## Lecturas recomendadas

- *High-Resolution Image Synthesis with Latent Diffusion
  Models* — Rombach et al., 2022.
- *SDXL* — Podell et al., 2023.
- *Adding Conditional Control to Text-to-Image Diffusion
  Models* — Zhang et al., 2023.
- *LoRA* — Hu et al., 2021.
- HuggingFace Diffusers: <https://huggingface.co/docs/diffusers>.

---

> 📚 **Adaptación al español** de la lección "[Latent Diffusion: Stable Diffusion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
