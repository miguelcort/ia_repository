# 05 — StyleGAN

> StyleGAN (Karras et al., 2019, 2020, 2021) es la familia de GANs que produce imágenes de alta calidad con control de estilo. StyleGAN3 iguala StyleGAN2 con técnicas para evitar artefactos y aliasing.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-gans-generador-y-discriminador
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el mapeo de estilo en StyleGAN (mapping
  network + AdaIN).
- Conocer las técnicas de StyleGAN2 (weight
  demodulation) y StyleGAN3 (alias-free).
- Diagnosticar cuándo StyleGAN es la mejor opción para
  generación de imágenes.
- Aplicar StyleGAN para tareas prácticas: inversion,
  editing, interpolation.

## El problema

DCGAN y ProGAN producen imágenes de calidad, pero con
control limitado. StyleGAN (Karras et al., 2019) introduce
un mapeo de estilo: un mapping network transforma z en
un espacio de estilo intermedio w, y AdaIN inyecta w
en cada capa del generador. StyleGAN2 corrige los
artefactos de StyleGAN1. StyleGAN3 introduce técnicas
"alias-free" para imágenes más naturales. La lección
cubre la familia y cuándo usarla.

## El concepto

**StyleGAN1 (2019).** Generador progresivo (de ProGAN)
con:

- **Mapping network:** z ∈ R^512 → w ∈ R^512. 8 capas
  MLP. Esto "desentrelaza" los atributos latentes.
- **AdaIN (Adaptive Instance Normalization):** normaliza
  feature maps y aplica w. Permite inyectar estilo a
  diferentes resoluciones.
- **Noise inputs:** ruido gaussiano por capa, para
  detalles estocásticos (cabellos, poros).
- **Progressive growing:** resolución 4×4 → 1024×1024
  con fade-in.
- **Style mixing:** dos w diferentes en diferentes
  capas para combinar estilos.

**StyleGAN2 (2020).** Corrige los artefactos "droplet"
de StyleGAN1 con:

- **Weight demodulation:** en vez de AdaIN, demodula
  pesos por capa. Más estable.
- **Path length regularization:** regulariza el camino
  en el espacio latente.
- **No progressive growing:** genera a resolución
  final directamente.
- **Lazy regularization:** aplica regularización cada
  N steps, no en cada step.

**StyleGAN3 (2021).** Elimina aliasing con:

- **Alias-free design:** todas las operaciones
  preservan el teorema de muestreo de Nyquist.
- **Fourier features:** representación en dominio
  frecuencia.
- **Filtered style mixing:** previene aliasing al
  combinar estilos.

**Cuándo usar StyleGAN.**

- Generación de caras, animales, texturas, arte.
- Inversión (encoder) para proyectos de un modelo
  pre-entrenado.
- Editing: mover w en direcciones específicas
  (edad, sonrisa, género).
- Style transfer y mixing.

**Cuándo NO usar StyleGAN.**

- Para imágenes médicas, científicas: requieren
  precisión. Usar diffusion models.
- Para texto: no aplica (usar transformers).
- Para 3D: usar NeRF o 3D-aware GANs.
- Para recursos limitados: StyleGAN es grande.
  Distilled variants o usar diffusion latente.

**Trampas.**

- **Mode collapse:** StyleGAN es menos propenso que
  DCGAN, pero aún posible. WGAN-GP, discriminator
  augmentation (ADA).
- **Dataset pequeño (< 30k):** usar ADA (Augmented
  Discriminator Augmentation) o transfer learning.
- **Sin inversion:** el espacio w no es trivial de
  invertir. Usar encoder (e4e, pSp) o métodos de
  optimización (Projected GAN).

## Constrúyelo

```python
import numpy as np


def mapping_network(z, weights, biases):
    """Mapping network de StyleGAN: z -> w.
    8 capas MLP con leaky ReLU."""
    h = z
    for W, b in zip(weights, biases):
        h = np.maximum(0.2 * h, h) @ W + b  # LeakyReLU
    return h


def adain(x, w, scale, shift):
    """Adaptive Instance Normalization.
    x: (N, C, H, W). w: estilo (C,).
    scale = affine(w). shift = affine(w)."""
    mean = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + 1e-8)
    return scale.reshape(-1, 1, 1) * x_norm + shift.reshape(-1, 1, 1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-stylegan
fase: 08
leccion: 05
---

Eres un asistente que ayuda con StyleGAN. Recibirás la
tarea y los datos. Tu trabajo:

1. Para caras, animales, texturas: StyleGAN2 o
   StyleGAN3.
2. Para edición: usar encoder pre-entrenado
   (e4e, pSp).
3. Para inversión: encoder > optimización directa
   (más rápido).
4. Para datasets pequeños (< 30k): ADA, transfer
   learning, o few-shot GAN.
5. Para edición controlada: interfazgans2, GANSpace,
   o StyleFlow.
6. Para evaluar: FID, PPL, slicing discovery.
7. Advertir contra mode collapse, aliasing, y overfitting.
```

## Ejercicios

1. **StyleGAN2**: pre-entrena en FFHQ usando el
   repositorio oficial.
2. **Inversión**: usa e4e para invertir una imagen
   al espacio w.
3. **Desafío**: edita la edad de una cara moviéndose
   en la dirección "age" del espacio w.

## Lecturas recomendadas

- *A Style-Based Generator Architecture for Generative
  Adversarial Networks* — Karras et al., 2019.
- *Analyzing and Improving the Image Quality of
  StyleGAN* — Karras et al., 2020.
- *Alias-Free Generative Adversarial Networks* — Karras
  et al., 2021.
- NVlabs/stylegan3: <https://github.com/NVlabs/stylegan3>.

---

> 📚 **Adaptación al español** de la lección "[StyleGAN]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
