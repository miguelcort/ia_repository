# 10 — Generación de imágenes con difusión

> Los modelos de difusión (DDPM, Ho et al., 2020) son el estado del arte en generación de imágenes. Stable Diffusion, DALL-E, Midjourney, Sora todos se basan en ellos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-generacion-de-imagenes-gan
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Implementar DDPM (Denoising Diffusion Probabilistic
  Models) desde cero.
- Aplicar el forward y reverse process.
- Diagnosticar hiperparámetros: número de pasos, schedule
  de ruido, beta.
- Conocer variantes: DDIM, score-based, latent diffusion.

## El problema

Las GANs tienen problemas: mode collapse, entrenamiento
inestable, difícil de escalar. Los modelos de difusión
ofrecen una alternativa: aprende a revertir un proceso de
ruido progresivo. Son más estables de entrenar, generan
imágenes de mayor calidad, y escalan a alta resolución
mejor. Stable Diffusion, DALL-E 3, Midjourney, Imagen —
todos son modelos de difusión.

## El concepto

**Forward process (diffusion).** Añade ruido gaussiano
progresivamente a una imagen real `x_0` hasta obtener
ruido puro `x_T`:

```text
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) · x_{t-1}, β_t · I)
q(x_t | x_0) = N(x_t; sqrt(α̅_t) · x_0, (1 - α̅_t) · I)
```

donde `β_t` es el schedule de ruido y `α̅_t = Π_{s=1..t} (1 -
β_s)`. Después de `T` pasos (típico 1000), `x_T` es
puro ruido gaussiano.

**Reverse process (denoising).** Aprende a invertir el
proceso. Una red neuronal predice el ruido `ε` añadido en
cada paso (o directamente `x_{t-1}`):

```text
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Loss de entrenamiento.** Simple y elegante:

```text
L = E[||ε - ε_θ(x_t, t)||²]
```

donde `ε` es el ruido real añadido y `ε_θ(x_t, t)` es la
predicción de la red. La red aprende a predecir el ruido,
no la imagen.

**Forward en forma cerrada.** `x_t = sqrt(α̅_t) · x_0 +
sqrt(1 - α̅_t) · ε` con `ε ~ N(0, I)`. Esto permite
entrenar en cualquier timestep sin simular los 1000 pasos.

**Arquitectura U-Net.** La red que predice el ruido es un
U-Net con atención. En cada nivel: ResNet block + downsample
(o upsample), y atención en las resoluciones más bajas.

**Schedule de ruido.** Controla cuánto ruido se añade en
cada paso. Variantes:

- **Linear:** `β_t = β_min + (β_max - β_min) * t / T`.
- **Cosine:** preserva SNR mejor; usado en Improved DDPM.
- **Scaled linear:** escala con resolución.

**Sampling.** En inferencia, partimos de `x_T ~ N(0, I)` y
aplicamos el reverse process `T` pasos. Cada paso invoca
la red neuronal, lo que es lento (1000 pasos = 1000
forward passes).

**Variantes de sampling para acelerar.**

- **DDIM (Denoising Diffusion Implicit Models):**
  sampling determinista, permite 50-100 pasos con
  calidad similar.
- **DPM-Solver:** ODE solver de alto orden.
- **Euler / Heun:** integradores clásicos aplicados a
  la ODE de difusión.

**Classifier-free guidance.** En difusión condicional
(texto → imagen), se entrena con y sin conditioning. En
inferencia, se mezcla:

```text
ε_guidance = ε_uncond + γ · (ε_cond - ε_uncond)
```

`γ > 1` amplifica la influencia del conditioning, dando
imágenes más alineadas con el prompt pero menos diversas.

**Latent diffusion (Stable Diffusion).** Aplica difusión
en el espacio latente del VAE en vez del espacio de
píxeles. La imagen se codifica al latente (4x downsampling),
la difusión opera en 64x64x4, y se decodifica al final.
Permite entrenar en GPUs de consumo (8 GB VRAM).

**Trampas.**

- **Schedule mal elegido:** imágenes oscuras (linear) o
  artefactos (cosine sin warmup).
- **Pocos pasos de sampling:** calidad degradada. Usar al
  menos 50 pasos con DDIM.
- **Guidance scale demasiado alto:** imágenes sobre-
  saturadas y con artefactos.

## Constrúyelo

```python
import numpy as np


def make_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = np.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def forward_diffusion(x0, t, alpha_bars):
    """Forward en forma cerrada: x_t = sqrt(α̅_t)·x_0 + sqrt(1-α̅_t)·ε."""
    eps = np.random.randn(*x0.shape)
    a_bar = alpha_bars[t]
    return np.sqrt(a_bar) * x0 + np.sqrt(1 - a_bar) * eps, eps


def reverse_step(x_t, t, eps_pred, betas, alphas, alpha_bars):
    """Un paso del reverse process."""
    beta_t = betas[t]
    a_t = alphas[t]
    a_bar_t = alpha_bars[t]
    mean = (1 / np.sqrt(a_t)) * (
        x_t - beta_t / np.sqrt(1 - a_bar_t) * eps_pred
    )
    if t > 0:
        return mean + np.sqrt(beta_t) * np.random.randn(*x_t.shape)
    return mean


def ddim_step(x_t, t, t_prev, eps_pred, alpha_bars):
    """DDIM: paso determinista."""
    a_t = alpha_bars[t]
    a_prev = alpha_bars[t_prev] if t_prev >= 0 else 1.0
    x0_pred = (x_t - np.sqrt(1 - a_t) * eps_pred) / np.sqrt(a_t)
    return np.sqrt(a_prev) * x0_pred + np.sqrt(1 - a_prev) * eps_pred
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-diffusion
fase: 04
leccion: 10
---

Eres un asistente que ayuda a entrenar un modelo de
difusión. Recibirás el dataset, la resolución, el hardware
y el tiempo de entrenamiento. Tu trabajo:

1. Si resolución <= 256x256 y GPU limitada: DDPM en
   espacio de píxeles.
2. Si resolución >= 512x512 o GPU con < 16GB: Latent
   Diffusion (Stable Diffusion).
3. Si quieres condicionamiento (texto, clase): Stable
   Diffusion con U-Net condicional.
4. Schedule: cosine en Improved DDPM; linear en DDPM
   clásico.
5. Entrenar 100k-1M steps.
6. AdamW con lr=1e-4.
7. Samplear con DDIM (50-100 pasos) o DPM-Solver
   (20-30 pasos).
8. Guidance scale 7.5 para Stable Diffusion.
9. Evaluar con FID sobre 10k muestras.
```

## Ejercicios

1. **DDPM desde cero**: implementa y entrena en MNIST.
2. **Schedule**: visualiza diferentes schedules de ruido
   y compara muestras.
3. **Desafío**: implementa classifier-free guidance
   sobre un dataset condicional simple.

## Lecturas recomendadas

- *Denoising Diffusion Probabilistic Models* — Ho et al.,
  2020.
- *Improved Denoising Diffusion Probabilistic Models* —
  Nichol & Dhariwal, 2021.
- *Denoising Diffusion Implicit Models* — Song et al.,
  2021.
- diffusers: <https://huggingface.co/docs/diffusers>.

---

> 📚 **Adaptación al español** de la lección "[Image Generation: Diffusion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
