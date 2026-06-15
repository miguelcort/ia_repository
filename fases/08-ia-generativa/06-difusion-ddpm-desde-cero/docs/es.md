# 06 — Difusión DDPM desde cero

> Los modelos de difusión (Ho et al., 2020) aprenden a revertir un proceso de ruido gaussiano. DDPM, DDIM, y score-based SDEs son la base de Stable Diffusion, DALL-E 2/3, e Imagen.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-cnn-basica
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Entender el forward (noise) y reverse (denoise) process.
- Implementar DDPM y DDIM.
- Conocer score-based SDEs y conexiones con NCSN.
- Diagnosticar cuándo usar difusión vs GANs.

## El problema

Los modelos de difusión (Ho et al., 2020) aprenden a
generar datos revirtiendo un proceso de ruido. Idea:
destruir gradualmente la estructura de los datos con
ruido gaussiano, y entrenar una red neuronal para
predecir y revertir ese ruido. La lección cubre
DDPM, DDIM, y score-based SDEs.

## El concepto

**Forward process (fixed).** q(x_t | x_{t-1}) =
N(x_t; sqrt(1 - β_t) x_{t-1}, β_t I). El schedule β_t
crece linealmente de 1e-4 a 0.02 en T=1000 steps.
Permite "saltar" a cualquier timestep:

```text
q(x_t | x_0) = N(x_t; sqrt(α̅_t) x_0, (1 - α̅_t) I)
```

donde α_t = 1 - β_t y α̅_t = ∏_{s=1..t} α_s.

**Reverse process (learned).** p_θ(x_{t-1} | x_t) =
N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t)). Se entrena para
predecir el ruido ε_θ(x_t, t) (reparameterization de
Ho et al.). La loss es simplemente MSE entre ε
predicho y ε real:

```text
L = E[ || ε - ε_θ(x_t, t) ||² ]
```

**Arquitectura.** U-Net (Ronneberger et al., 2015) con
residual blocks, self-attention en bajas resoluciones, y
time embedding sinusoidal. Típica 100M-1B parámetros.

**Sampling.**

- **DDPM (Ho et al., 2020):** ancestral sampling, T=1000
  steps. Lento pero buena calidad.
- **DDIM (Song et al., 2020):** deterministic sampling,
  permite T=50-100 steps. Más rápido, similar calidad.
- **DPM-Solver (Lu et al., 2022):** orden superior, T=10-
  20 steps.
- **Score-based SDEs (Song et al., 2021):** marco general
  con SDEs, Euler-Maruyama y predictor-corrector.

**Conditional generation.**

- **Classifier guidance:** entrenar classifier p(y|x_t)
  y usar su gradiente.
- **Classifier-free guidance:** entrenar modelo condicional
  y no-condicional, combinar en inference.
- **CFG scale:** 7-15 es típico. Mayor = más adherence al
  prompt, menos diversity.

**Cuándo usar difusión.**

- **Imágenes de alta calidad:** Stable Diffusion, DALL-E,
  Imagen.
- **Audio:** WaveGrad, DiffWave.
- **3D:** diffusion sobre NeRF, point clouds.
- **Texto:** más difícil; usar transformers
  autoregresivos.

**Cuándo NO usar difusión.**

- **Real-time generation:** GANs son más rápidos
  (1 forward pass).
- **Mode coverage perfecto:** difusión es mejor, pero
  si necesitas log-likelihood, usar flows.
- **Recursos limitados:** difusión es lento. Usar GANs,
  VAE, o modelos destilados.

**Trampas.**

- **Sin normalización:** datos deben estar en [-1, 1]
  para que la red prediga ε en la misma escala.
- **Sin time embedding:** la red no sabe en qué timestep
  está. Usar sinusoidal embeddings o FiLM.
- **Muestreo con T muy bajo:** degrada calidad. Usar
  DPM-Solver o score-based.

## Constrúyelo

```python
import numpy as np


def forward_diffusion(x_0, t, beta_t, alpha_bar_t):
    """q(x_t | x_0) = N(sqrt(α̅_t) x_0, (1 - α̅_t) I)."""
    eps = np.random.randn(*x_0.shape)
    mean = np.sqrt(alpha_bar_t[t]) * x_0
    var = 1 - alpha_bar_t[t]
    return mean + np.sqrt(var) * eps, eps


def ddpm_loss(eps_true, eps_pred):
    """DDPM loss: MSE entre ruido real y predicho."""
    return np.mean((eps_true - eps_pred) ** 2)


def ddpm_sample(x_T, eps_pred_fn, alpha, alpha_bar, beta,
               T=1000):
    """Ancestral sampling: x_T -> x_0."""
    x = x_T
    for t in reversed(range(T)):
        eps = eps_pred_fn(x, t)
        mean = (1 / np.sqrt(alpha[t])) * (x
                - beta[t] / np.sqrt(1 - alpha_bar[t]) * eps)
        if t > 0:
            x = mean + np.sqrt(beta[t]) * np.random.randn(*x.shape)
        else:
            x = mean
    return x
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
fase: 08
leccion: 06
---

Eres un asistente que ayuda con modelos de difusión.
Recibirás la tarea y los datos. Tu trabajo:

1. Para imágenes: Stable Diffusion, DALL-E, Imagen.
2. Para real-time: usar destilación (LCM, SDXL-Turbo)
   o GANs.
3. Para datos propios: entrenar DDPM o DDIM desde
   cero.
4. Para muestreo rápido: DPM-Solver, T=10-20.
5. Para condicional: classifier-free guidance, scale=7-15.
6. Para evaluar: FID, IS, CLIP score.
7. Advertir contra muestreo con T muy bajo, sin time
   embedding, y datos sin normalizar.
```

## Ejercicios

1. **Mini-DDPM**: implementa y entrena en MNIST.
2. **DDIM sampler**: implementa y compara con DDPM.
3. **Desafío**: entrena un modelo de difusión
   condicional en CIFAR-10.

## Lecturas recomendadas

- *Denoising Diffusion Probabilistic Models* — Ho et
  al., 2020.
- *Denoising Diffusion Implicit Models* — Song et al.,
  2020.
- *Score-Based Generative Modeling through SDEs* — Song
  et al., 2021.
- *DPM-Solver* — Lu et al., 2022.
- Lilian Weng's blog: <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/>.

---

> 📚 **Adaptación al español** de la lección "[Diffusion DDPM from Scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
