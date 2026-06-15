# 28 — Modelos del mundo y difusión de video

> Los modelos del mundo (world models) aprenden a simular la física y dinámica de un entorno. Sora, Veo, y Wan son modelos del mundo para video.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 23-difusion-transformers-y-flujo-rectificado,
                  12-comprension-de-video
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el concepto de "modelo del mundo" (world
  model).
- Implementar video diffusion con DiT.
- Aplicar Sora, Veo, Wan, y modelos open-source.
- Diagnosticar consistencia temporal y física.

## El problema

Los modelos de difusión producen imágenes individuales
impresionantes, pero video requiere **consistencia
temporal**: un objeto que se mueve en frame 1 debe estar
en la posición esperada en frame 2. Los **modelos del
mundo** van más allá: aprenden la física y dinámica de un
entorno, permitiendo generar video plausible. Sora
(OpenAI, 2024), Veo (Google, 2024), Wan (Alibaba, 2025)
son SOTA. La lección cubre la arquitectura y los
desafíos únicos del video.

## El concepto

**¿Qué es un modelo del mundo?** Un modelo que aprende la
dinámica `p(s_{t+1} | s_t, a_t)` donde `s` es el estado del
mundo y `a` es la acción del agente. En visión: `s` es un
frame de video y "predecir el siguiente frame" obliga al
modelo a aprender física, óptica, y causalidad. Haiku
(David Ha, 2018) y Dreamer (Hafner et al., 2020)
introdujeron el concepto para RL.

**Diffusion transformers para video.**

- **Patch embedding 3D:** el video `(T, H, W, C)` se divide
  en `(T/pt, H/ph, W/pw, pt*ph*pw*C)` patches
  espacio-temporales.
- **Positional embedding 3D:** combina timestep y
  posición espacial.
- **DiT blocks:** self-attention dentro de patches
  espaciales + temporales (factorized attention).
- **Decoder:** proyecta patches 3D a video `(T, H, W, C)`.

**Sora (OpenAI, 2024).** DiT con spatial + temporal
attention factorizada. Genera videos de hasta 60
segundos a 1080p. El paper "Video generation models as
world simulators" (Brooks et al., 2024) revela que es
básicamente un DiT con patchification espacio-temporal.

**Veo (Google, 2024).** Modelo cerrado de Google.
Calidad SOTA en 2024, mejor coherencia temporal que
Sora 1 según benchmarks internos.

**Wan 2.1 (Alibaba, 2025).** Open-source SOTA en
video generation. DiT + flow matching, hasta 1080p y
5 segundos. License Apache 2.0.

**Desafíos únicos del video.**

- **Consistencia temporal:** flickering, morphing,
  identidades que cambian.
- **Coherencia física:** objetos que flotan, se
  teletransportan, o无视 la gravedad.
- **Costo computacional:** un video de 60 segundos a
  1080p son 30M de píxeles. La memoria de atención
  es `O((T*H*W)²)`. La factorización es esencial.
- **Datos:** video de alta calidad es escaso. La
  curación (como LAION-5B para imágenes) es costosa.

**Métricas de evaluación.**

- **FVD (Fréchet Video Distance):** análogo temporal de
  FID.
- **CLIP score:** adherencia al prompt (como en
  imágenes).
- **Human eval:** comparación lado a lado por
  humanos.
- **Consistencia temporal:** flujo óptico entre
  frames consecutivos.

**Trampas.**

- **Training cost:** los modelos del mundo
  requieren miles de GPU-días. No se entrenan desde
  cero a menos que seas OpenAI.
- **Prompts físicos:** los modelos no entienden
  gravedad, fricción, ni causalidad como un LLM
  entiende gramática. Fallan en física contraintuitiva.
- **Consistencia a largo plazo:** los modelos
  pierden coherencia después de 5-10 segundos.

## Constrúyelo

```python
import numpy as np


def patchify_video(video, patch_t=2, patch_h=16, patch_w=16):
    """Divide video (T, H, W, C) en patches espacio-temporales.
    Devuelve (N, patch_t * patch_h * patch_w * C)."""
    T, H, W, C = video.shape
    n_t = T // patch_t
    n_h = H // patch_h
    n_w = W // patch_w
    # Simplificado: reshape y transpose
    return video[:n_t * patch_t, :n_h * patch_h, :n_w * patch_w, :].reshape(
        n_t, n_h, n_w, patch_t, patch_h, patch_w, C
    ).reshape(n_t * n_h * n_w, -1)


def temporal_3d_positional_encoding(n_t, n_h, n_w, dim):
    """Encoding posicional 3D: combina tiempo, alto, ancho."""
    pe = np.zeros((n_t * n_h * n_w, dim))
    for t in range(n_t):
        for h in range(n_h):
            for w in range(n_w):
                idx = (t * n_h + h) * n_w + w
                for d in range(dim // 3):
                    pe[idx, d] = np.sin(t / (10000 ** (2 * d / dim)))
                    pe[idx, dim // 3 + d] = np.sin(h / (10000 ** (2 * d / dim)))
                    pe[idx, 2 * dim // 3 + d] = np.sin(w / (10000 ** (2 * d / dim)))
    return pe
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-world-model
fase: 04
leccion: 28
---

Eres un asistente que ayuda a generar video con modelos
del mundo. Recibirás el prompt y la duración objetivo. Tu
trabajo:

1. Si quieres SOTA hospedado: Sora (OpenAI) o Veo
   (Google).
2. Si quieres open-source: Wan 2.1 o CogVideoX.
3. Si quieres control fino: entrenar DiT para video
   con tu dataset.
4. Resolución: 480p es lo más manejable; 1080p
   requiere mucho cómputo.
5. Duración: 5-10 segundos es realista. Más
   requiere consistencia temporal reforzada.
6. Prompts: descriptivos, con movimiento explícito.
7. Evaluar: FVD, CLIP score, consistencia temporal.
8. Postprocesar: interpolación de frames para
   suavizar.
9. Advertir: la física a menudo no se respeta.
```

## Ejercicios

1. **Patchify video**: implementa la patchificación
   espacio-temporal.
2. **Positional encoding 3D**: implementa el encoding
   combinado tiempo+espacio.
3. **Desafío**: entrena un DiT-video pequeño en un
   dataset de video.

## Lecturas recomendadas

- *Video generation models as world simulators* —
  Brooks et al., 2024 (Sora technical report).
- *Wan: Open and Advanced Large-Scale Video Generative
  Models* — Alibaba, 2025.
- *CogVideoX* — Zhipu, 2024.
- DiT: <https://github.com/facebookresearch/DiT>.

---

> 📚 **Adaptación al español** de la lección "[World Models and Video Diffusion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
