# 23 — Diffusion transformers y rectified flow

> DiT (Diffusion Transformer) reemplaza el U-Net del UNet con un transformer. Stable Diffusion 3, FLUX.1, Sora todos usan DiT. Rectified Flow es la nueva ODE que reemplaza la SDE clásica.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-generacion-de-imagenes-con-difusion
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender la arquitectura DiT y por qué reemplaza al U-Net.
- Implementar Rectified Flow y entender la diferencia con
  DDPM.
- Diagnosticar trade-offs: calidad vs costo de
  entrenamiento.
- Conocer Sora, FLUX.1, Stable Diffusion 3.

## El problema

El U-Net con cross-attention dominó la generación de
imágenes con difusión desde DDPM (2020). Pero los
transformers han ganado en NLP y visión (ViT, Swin). Los
**Diffusion Transformers (DiT, Peebles & Xie, 2023)**
reemplazan el U-Net con un transformer puro (como ViT).
Stable Diffusion 3, FLUX.1, y Sora todos usan DiT.
**Rectified Flow** (Liu et al., 2022) ofrece una nueva
formulación que permite flujos rectos (en vez del
brownian motion curvo), permitiendo muestreo más rápido
con menos pasos.

## El concepto

**DiT architecture.**

- **Patch embedding:** igual que ViT. Divide la imagen
  (o el latente en latent diffusion) en patches y los
  proyecta.
- **Positional embedding:** aprendible, 2D para
  imágenes, 3D para video.
- **AdaLN-Zero (Adaptive Layer Norm):** la condicionante
  (timestep, class label, text embedding) se inyecta vía
  layer norm adaptativo, no cross-attention. Más simple y
  más eficiente.
- **Transformer blocks:** estándar, multi-head
  self-attention + MLP.
- **Final layer:** patch → pixel (o latente).

**Ventajas DiT vs U-Net.**

- Mejor escalabilidad: scaling laws más limpias.
- Sin inductive bias: aprende la convolución si la
  necesita.
- Más fácil de paralelizar.
- Calidad SOTA en benchmarks.

**Rectified Flow.** En vez de añadir ruido browniano
curvo al training data, **rectified flow** aprende un
flujo recto entre `x_0` (data) y `x_1` (ruido). El ODE
resultante es:

```text
dx/dt = v_θ(x_t, t)
```

donde `v_θ` predice la dirección del flujo. Entrenar es
regression sobre la diferencia `x_1 - x_0`:

```text
L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

donde `x_t = (1 - t) · x_0 + t · x_1` (interpolación
lineal). Esto da trayectorias rectas, permitiendo
sampling con muy pocos pasos (10-20).

**Stable Diffusion 3 (Esser et al., 2024).** DiT +
rectified flow + MMDiT (multimodal DiT que combina texto e
imagen en la arquitectura). SOTA en text-to-image
open-source en 2024.

**FLUX.1 (Black Forest Labs, 2024).** DiT + flow
matching + parallel attention. SOTA en fidelidad y
adherencia al prompt en 2024-2025.

**Sora (OpenAI, 2024).** DiT para video. Diffusion
transformer conpositional (espacial + temporal). Genera
videos de hasta 60 segundos a 1080p.

**Cuándo usar DiT vs U-Net.**

| Aspecto | U-Net | DiT |
|---|---|---|
| Training cost | Bajo | Alto (necesita scale) |
| Calidad SOTA | Buena | Mejor |
| Sampling | DDIM/DPM | Flow matching |
| Adopción | Legacy | SOTA 2024-2025 |

**Trampas.**

- **Training cost:** DiT necesita datasets grandes
  (millones de imágenes) y mucho cómputo (cientos de
  GPU-días).
- **Sin AdaLN:** las primeras versiones de DiT usaban
  cross-attention. AdaLN-Zero es la versión moderna.
- **Mala inicialización:** DiT necesita warmup largo
  y lr bajo.

## Constrúyelo

```python
import numpy as np


def rectified_flow_step(x_t, t, v_pred, dt):
    """Un paso del ODE de rectified flow."""
    return x_t + v_pred * dt


def rectified_flow_loss(x_0, x_1, v_pred):
    """L2 entre la velocidad predicha y la dirección real."""
    return float(np.mean((v_pred - (x_1 - x_0)) ** 2))


def adaln(gamma, beta, x):
    """Adaptive Layer Norm: modula x con gamma, beta."""
    return gamma * (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-6
    ) + beta


def simple_dit_block(x, t_emb, c_emb, W_qkv, W_mlp):
    """Bloque DiT con AdaLN-Zero simplificado."""
    # AdaLN: gamma y beta derivados de (t, c)
    gamma, beta = c_emb[..., :x.shape[-1]], c_emb[..., x.shape[-1]:]
    x = adaln(gamma, beta, x)
    # Self-attention (simplificado)
    q, k, v = x @ W_qkv[:x.shape[-1]], x @ W_qkv[
        x.shape[-1]:2 * x.shape[-1]
    ], x @ W_qkv[2 * x.shape[-1]:]
    attn = np.exp(q @ k.T - (q @ k.T).max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    x = x + attn @ v
    # MLP
    gamma2, beta2 = c_emb[..., :x.shape[-1]], c_emb[..., x.shape[-1]:]
    x = adaln(gamma2, beta2, x)
    h = np.maximum(0, x @ W_mlp[0])
    x = x + h @ W_mlp[1]
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
name: prompt-dit
fase: 04
leccion: 23
---

Eres un asistente que ayuda a usar DiT o Rectified Flow.
Recibirás el dataset, el hardware y la calidad objetivo.
Tu trabajo:

1. Si quieres SOTA: usar Stable Diffusion 3, FLUX.1 o
   Sora (vía API).
2. Si quieres entrenar desde cero: DiT-B/2 o DiT-XL/2
   con flow matching.
3. Si quieres text-to-image: usar CLIP text encoder
   para condicionar.
4. Si quieres video: DiT con positional embedding 3D
   y frame interpolation.
5. Training cost: DiT-XL necesita 256 GPUs A100 por ~7
   días. Usa modelos preentrenados.
6. Sampling: rectified flow con 10-25 pasos.
7. Guidance scale: 3-7.
8. Evaluar con FID, CLIP score, human eval.
```

## Ejercicios

1. **Rectified flow**: implementa el sampling y compara
   con DDPM.
2. **AdaLN-Zero**: implementa el bloque DiT con
   AdaLN-Zero.
3. **Desafío**: entrena un DiT pequeño en CIFAR-10.

## Lecturas recomendadas

- *Scalable Diffusion Models with Transformers (DiT)* —
  Peebles & Xie, 2023.
- *Flow Matching for Generative Modeling* — Lipman et
  al., 2023.
- *Scaling Rectified Flow Transformers for High-Resolution
  Image Synthesis* — FLUX.
- diffusers: <https://huggingface.co/docs/diffusers>.

---

> 📚 **Adaptación al español** de la lección "[Diffusion Transformers and Rectified Flow]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
