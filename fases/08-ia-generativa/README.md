# Fase 8 — IA generativa

> Crear imágenes, video, audio, 3D y más.

Los **modelos generativos** son la cara más visible de la IA en
2026. Crean imágenes (Stable Diffusion, FLUX, DALL-E 3), video
(Sora, Veo, Runway Gen-3), audio (MusicGen, Bark), objetos 3D
(DreamFusion, Gaussian Splatting) y texto (GPT, Claude, Gemini).
Esta fase recorre el espectro generativo con la misma filosofía
del currículo: **construir antes de usar**. Empezamos con
autoencoders, VAEs y GANs, llegamos a difusión desde cero, y luego
subimos el nivel hasta hacer *fine-tuning* de Stable Diffusion con
LoRA, edición con ControlNet, y evaluación con FID/CLIP score.

La fase tiene **cuatro bloques**. El **bloque 1** (lecciones 1–2)
introduce la taxonomía: autoencoders y VAEs. El **bloque 2** (3–5)
cubre **GANs**: vanilla, condicionales (Pix2Pix) y StyleGAN. El
**bloque 3** (6–13) es **difusión y nuevas familias**: DDPM desde
cero, Stable Diffusion, ControlNet, LoRA, inpainting, video,
audio, 3D y flow matching. El **bloque 4** (14–16) cierra con
**evaluación y modelos autorregresivos visuales**: FID, CLIP score
y VAR.

## Índice de lecciones

### Bloque 1 — Taxonomía y representación latente

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Taxonomía e historia de modelos generativos](01-taxonomia-y-historia-de-modelos-generativos/) | Aprender | VAEs, GANs, difusión, autorregresivos, flow. |
| 02 | [Autoencoders y VAE](02-autoencoders-y-vae/) | Construir | Encoder-decoder, *reparameterization trick* y ELBO. |

### Bloque 2 — GANs

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 03 | [GANs: generador y discriminador](03-gans-generador-y-discriminador/) | Construir | *Min-max game*, *mode collapse* y *Wasserstein loss*. |
| 04 | [GANs condicionales y Pix2Pix](04-gans-condicionales-pix2pix/) | Construir | *Conditional batch norm*, *U-Net* generador, *patch* discriminator. |
| 05 | [StyleGAN](05-stylegan/) | Construir | *Mapping network*, *style mixing* y *progressive growing*. |

### Bloque 3 — Difusión y familias modernas

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [DDPM desde cero](06-difusion-ddpm-desde-cero/) | Construir | *Forward/reverse process*, *noise schedule*, ε-prediction. |
| 07 | [Stable Diffusion y difusión latente](07-difusion-latente-stable-diffusion/) | Construir | VAE + U-Net + CLIP, *text conditioning*. |
| 08 | [ControlNet, LoRA y condicionamiento](08-controlnet-y-lora-condicionamiento/) | Construir | *Fine-tuning* eficiente con bajo costo de memoria. |
| 09 | [Inpainting, outpainting y edición](09-inpainting-outpainting-y-editing/) | Construir | Máscaras, *latent blending* y *prompt-to-prompt*. |
| 10 | [Generación de video](10-generacion-de-video/) | Construir | Sora, Veo, *latent video diffusion*, *temporal attention*. |
| 11 | [Generación de audio](11-generacion-de-audio/) | Construir | AudioLDM, MusicGen, *neural codec* conditioning. |
| 12 | [Generación 3D](12-generacion-3d/) | Construir | DreamFusion, *score distillation*, *Gaussian splatting*. |
| 13 | [Flow matching y rectified flows](13-flow-matching-y-rectified-flows/) | Construir | ODE/SDE, transporte óptimo, Stable Diffusion 3. |

### Bloque 4 — Evaluación y modelos autorregresivos

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 14 | [Evaluación: FID y CLIP score](14-evaluacion-fid-y-clip-score/) | Construir | Métricas estándar para modelos de imagen. |
| 15 | [Modelado autorregresivo visual (VAR)](15-visual-autoregressive-var/) | Construir | *Next-scale prediction*, alternativa a la difusión. |
| 16 | [Evaluación de modelos generativos](16-evaluacion-de-modelos-generativos/) | Construir | *Human eval*, CLIP score, FAD, *diversity metrics*. |

## Prerrequisitos

- **Fases 0, 1, 3, 4 y 7** completas.
- Conocimiento sólido de PyTorch y de arquitecturas encoder-
  decoder.
- GPU con 12+ GB VRAM (16+ recomendado para *fine-tuning* de
  Stable Diffusion con LoRA).
- Opcional: familiaridad con la **Fase 6 (audio)** para la
  lección 11.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Implementar** un VAE y un DDPM en MNIST desde cero.
- **Construir** una GAN y entrenarla establemente, diagnosticando
  *mode collapse* y problemas de balance.
- **Hacer fine-tuning** de Stable Diffusion con LoRA en un
  dominio personalizado.
- **Editar** imágenes con ControlNet, inpainting y *prompt-to-
  prompt*.
- **Evaluar** modelos generativos con FID, CLIP score, IS y
  diversidad.
- **Comprender** los nuevos paradigmas: flow matching, VAR,
  diffusion transformers, generación de video y 3D.
- **Seleccionar** la familia generativa adecuada para una tarea
  dada (imagen, video, audio, 3D).

## Stack y herramientas

- **PyTorch** y **torchvision**.
- **diffusers** de Hugging Face para Stable Diffusion.
- **transformers** para CLIP, T5, etc.
- **accelerate** y **peft** para *fine-tuning* eficiente.
- **ControlNet** y **T2I-Adapter** para condicionamiento.
- **trimesh** y **nerfstudio** para 3D.
- **torch-fidelity** y **clean-fid** para evaluación.
- **wandb** para tracking de experimentos.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **VAE** | Lección 02 | Stable Diffusion (VAE encoder/decoder). |
| **GAN** | Lecciones 3–5 | Predecesor de la difusión. |
| **DDPM** | Lección 06 | Base de Stable Diffusion, Sora. |
| **Latent diffusion** | Lección 07 | Estándar actual. |
| **LoRA** | Lección 08 | Fase 10 (LLM), Fase 11 (fine-tuning). |
| **ControlNet** | Lección 08 | Edición controlada. |
| **Flow matching** | Lección 13 | SD3, FLUX. |
| **FID** | Lección 14 | Métrica estándar. |
| **CLIP score** | Lección 14 | Métrica de texto-imagen. |

## Cómo estudiar esta fase

1. **La lección 06 (DDPM desde cero) es el corazón de la fase.**
   Sin entender el *forward/reverse process* y la predicción de
   ruido ε, el resto es caja negra.
2. **Usa datasets pequeños al principio** (MNIST, Fashion-MNIST,
   CelebA 64x64). Saltar a Stable Diffusion con SDXL frustra.
3. **Para *fine-tuning* de Stable Diffusion, empieza con un LoRA
   pequeño** (rank 16, 100 imágenes, 1000 pasos). Es la forma
   más rápida de ver resultados.
4. **No escales a video o 3D hasta dominar imagen.** Las
   lecciones 10 y 12 son intensivas en GPU; primero
   familiarízate con el pipeline de Stable Diffusion.
5. **Las métricas importan más que el resultado visual.** Una FID
   de 5 es excelente; una de 50 es mediocre. Aprende a calcular
   FID en cualquier modelo que entrenes.

## Verificación de progreso

```bash
# Lección 02 — VAE en MNIST
python3 fases/08-ia-generativa/02-autoencoders-y-vae/code/main.py

# Lección 06 — DDPM desde cero
python3 fases/08-ia-generativa/06-difusion-ddpm-desde-cero/code/main.py

# Lección 14 — FID/CLIP score
python3 fases/08-ia-generativa/14-evaluacion-fid-y-clip-score/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo usar cada familia generativa

| Caso de uso | Familia | Lección |
|---|---|---|
| Imágenes de alta calidad | Stable Diffusion, FLUX | 07 |
| Edición controlada de imagen | ControlNet, LoRA | 08 |
| Inpainting / outpainting | Stable Diffusion inpainting | 09 |
| Video | Sora, Veo, Wan | 10 |
| Música | MusicGen, AudioLDM | 11 |
| 3D a partir de texto | DreamFusion, 3DGS | 12 |
| Necesidad de ODE/SDE estable | Flow matching, RF | 13 |

## Conexión con otras fases

- **Entrada** → [Fase 7 — Transformers a fondo](../07-transformers-a-fondo/README.md)
  y [Fase 4 — Visión](../04-vision-por-computador/README.md).
- **Salida natural** → [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md)
  (muchas técnicas — LoRA, RLHF, evaluación — se comparten).
- **Reuso en** → Fase 6 (audio), Fase 12 (multimodal), Fase 19 (capstone).

## Recursos recomendados

- *Generative Deep Learning* — David Foster.
- *Denoising Diffusion Probabilistic Models* — Ho et al., 2020.
- *High-Resolution Image Synthesis with Latent Diffusion Models* — Rombach et al.
- *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al.
- *Adding Conditional Control to Text-to-Image Models* — Zhang et al. (ControlNet).
- *Hugging Face Diffusers* — <https://huggingface.co/docs/diffusers>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *diffusion*,
  *VAE*, *LoRA*, *FID*, *CLIP score*.
- [Fase 7 — Transformers a fondo](../07-transformers-a-fondo/README.md).
- [Fase 12 — IA multimodal](../12-ia-multimodal/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
