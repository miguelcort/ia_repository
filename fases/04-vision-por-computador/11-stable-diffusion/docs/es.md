# 11 — Stable Diffusion

> Stable Diffusion es el modelo de difusión open-source más popular. Combina VAE + UNet + CLIP text encoder para generar imágenes de alta calidad condicionadas por texto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-generacion-de-imagenes-con-difusion
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Entender los tres componentes: VAE, U-Net, CLIP text
  encoder.
- Usar Hugging Face diffusers para generación condicionada
  por texto.
- Implementar inpainting, outpainting e image-to-image.
- Diagnosticar trade-offs entre resolución y VRAM.

## El problema

Stable Diffusion (Rombach et al., 2022) democratizó la
generación de imágenes: puedes correr un modelo de
difusión en una GPU de 8GB. La idea clave es la **latent
diffusion**: aplica la difusión en el espacio latente de
un VAE (64x64x4) en vez del espacio de píxeles (512x512x3).
El VAE reduce dimensionalidad, el UNet predice ruido en
el latente, y CLIP convierte texto a embeddings que
condicionan al UNet.

## El concepto

**Los tres componentes.**

1. **VAE (Variational AutoEncoder):** codifica la imagen
   512x512x3 a un latente 64x64x4. Reduce dimensionalidad
   48x. El UNet opera en este espacio; el VAE decodifica
   al final.
2. **U-Net con cross-attention:** predice el ruido en
   el espacio latente. Tiene bloques ResNet + cross-
   attention a los embeddings de texto (de CLIP).
3. **CLIP text encoder:** convierte el prompt a un vector
   por token (77 tokens, 768 dim). El cross-attention del
   UNet lo consume en cada nivel.

**Pipeline de inferencia.**

1. Tokenizar el prompt: "un gato sentado en una silla" →
   77 tokens CLIP.
2. Codificar con CLIP: 77x768 embeddings.
3. Samplear `z_T ~ N(0, I)` con shape 64x64x4.
4. Aplicar 50-100 pasos del reverse process con el UNet,
   condicionado por los embeddings CLIP.
5. Decodificar el latente con el VAE: 64x64x4 → 512x512x3.

**Scheduler.** El planificador de ruido controla cómo se
añade y quita ruido. Variantes:

- **DDIM:** determinista, 50-100 pasos.
- **PNDM:** pseudo-numerical methods, 20-50 pasos.
- **DPM-Solver:** alto orden, 15-30 pasos.
- **Euler / Heun:** simples, 25-50 pasos.

**Guidance scale.** Classifier-free guidance:

```text
eps = eps_uncond + γ · (eps_cond - eps_uncond)
```

`γ = 7.5` es default. Más alto = más adherencia al prompt,
menos diversidad, posibles artefactos.

**Negative prompt.** "Lo que NO quieres". Usa unconditional
guidance con un prompt negativo. Útil para evitar manos
extrañas, artefactos, etc.

**Variantes de SD.**

- **SD 1.5:** original, 512x512, 860M parámetros UNet.
- **SD 2.x:** más datos, mejor calidad, OpenCLIP.
- **SDXL:** SD + refiner, 1024x1024, 2.6B parámetros.
- **SD3:** arquitectura MM-DiT (transformer en vez de UNet).
- **FLUX.1:** estado del arte 2024-2025, transformer-based.
- **SDXL-Turbo / SD-Turbo:** distillation a 1-4 pasos.

**Operaciones comunes.**

- **txt2img:** texto → imagen.
- **img2img:** imagen + texto → imagen modificada.
- **inpaint:** imagen + máscara + texto → región rellena.
- **outpaint:** extiende la imagen más allá del borde.
- **ControlNet:** condiciona con edge maps, depth, pose.
- **LoRA:** fine-tuning eficiente con rank decomposition.
- **IP-Adapter:** condiciona con imagen de referencia.

**Trampas.**

- **VRAM insuficiente:** reducir resolución a 512x512 o
  usar SDXL-Turbo.
- **Guidance muy alto:** artefactos y sobre-saturación.
- **Sampler muy lento:** usar DPM-Solver con 20 pasos.
- **LoRA mal entrenada:** degrada la calidad. Usar rank
  16-64, 1000-5000 steps.

## Constrúyelo

```python
import numpy as np


def latent_diffusion_step(z_t, t, eps_pred, alpha_bars,
                          guidance_scale=7.5):
    """Un paso de latent diffusion con CFG."""
    if guidance_scale > 1:
        # En producción: eps_uncond + γ · (eps_cond - eps_uncond)
        # Aquí simplificamos asumiendo eps_pred ya incluye CFG
        pass
    a_t = alpha_bars[t]
    z0_pred = (z_t - np.sqrt(1 - a_t) * eps_pred) / np.sqrt(a_t)
    return z0_pred


def vae_decode(z, decoder_weights):
    """Decodifica el latente 64x64x4 a 512x512x3.
    Simplificado: convs transpuestas."""
    h = z
    for w in decoder_weights:
        h = np.dot(h, w.T)  # conv transpuesta como matmul
        h = np.maximum(0, h)  # ReLU
    h = np.tanh(h)  # salida en [-1, 1]
    return (h + 1) / 2  # normalizar a [0, 1]
```

## Úsalo

```bash
pip install diffusers transformers accelerate
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-stable-diffusion
fase: 04
leccion: 11
---

Eres un asistente que ayuda a generar imágenes con
Stable Diffusion. Recibirás el prompt, el hardware y la
calidad objetivo. Tu trabajo:

1. Si GPU 8GB: SD 1.5, 512x512, fp16.
2. Si GPU 16GB: SDXL, 1024x1024.
3. Si quieres SOTA: FLUX.1 o SD3.
4. Si quieres 1-4 pasos: SDXL-Turbo o LCM-LoRA.
5. Sampler: DPM-Solver con 20-30 pasos.
6. Guidance: 7.5 para SD 1.5, 5.0 para SDXL.
7. Negative prompt: "blurry, low quality, distorted".
8. Para fine-tuning: LoRA con rank 16-64.
9. Para ControlNet: edge maps, depth, pose.
10. Inpainting: máscara + texto, denoise_strength 0.7.
```

## Ejercicios

1. **txt2img**: genera imágenes con diffusers usando
   distintos prompts y observa el efecto del guidance
   scale.
2. **img2img**: modifica una imagen existente con un
   prompt.
3. **Desafío**: fine-tunea SD 1.5 con LoRA en un
   dataset de tu elección.

## Lecturas recomendadas

- *High-Resolution Image Synthesis with Latent Diffusion
  Models* — Rombach et al., 2022.
- *Stable Diffusion 3* — Esser et al., 2024.
- *FLUX.1* — Black Forest Labs, 2024.
- diffusers: <https://huggingface.co/docs/diffusers>.
- AUTOMATIC1111: <https://github.com/AUTOMATIC1111/stable-diffusion-webui>.

---

> 📚 **Adaptación al español** de la lección "[Stable Diffusion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
