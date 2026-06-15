# 03 — GANs: generador y discriminador

> Las GANs (Goodfellow et al., 2014) entrenan dos redes en un juego adversarial: el generador crea muestras falsas, el discriminador intenta distinguirlas. El resultado: un generador capaz de producir imágenes, audio, o texto realistas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-cnn-basica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el juego adversarial entre generador y
  discriminador.
- Implementar la loss function de minimax.
- Conocer problemas comunes: mode collapse, training
  instability.
- Diagnosticar cuándo GANs son la mejor opción y cuándo
  no.

## El problema

Las GANs (Generative Adversarial Networks, Goodfellow
et al., 2014) entrenan dos redes en un juego min-max:

- **Generador G:** produce muestras falsas a partir de
  ruido z ~ p(z).
- **Discriminador D:** distingue muestras reales de
  falsas.

G y D se entrenan alternadamente. Idealmente, en el
equilibrio, G produce muestras indistinguibles de las
reales, y D no puede hacerlo mejor que adivinar. La
lección implementa GAN canónica y cubre variantes
modernas.

## El concepto

**Objetivo adversarial (original).**

```text
min_G max_D V(D, G) = E_{x ~ p_data(x)}[log D(x)]
                    + E_{z ~ p_z(z)}[log(1 - D(G(z)))]
```

G quiere minimizar; D quiere maximizar. En la práctica,
G se entrena con `max log D(G(z))` (en vez de min
log(1 - D(G(z)))), porque el gradiente es más fuerte
al inicio.

**Arquitectura típica.**

- **G:** input z (ruido, dim=100) → transposed conv
  (también llamada deconv) → imagen. Típico 4-5 capas
  con BN y ReLU. Tanh al final.
- **D:** imagen → strided conv (en vez de pooling) → 
  logit. Típico 4-5 capas con BN y LeakyReLU. Sigmoid
  para probabilidad.

**Problemas comunes.**

- **Mode collapse:** G produce solo unas pocas muestras
  diversas. D no las distingue, G no se ve forzado a
  diversificar. Solución: unrolled GANs, minibatch
  discrimination, WGAN.
- **Training instability:** loss de D y G oscila.
  Solución: TTUR (Two Time-Scale Update Rule), gradient
  penalty, spectral normalization.
- **Vanishing gradients:** si D es perfecto, G no recibe
  gradiente. Solución: usar Wasserstein loss
  (WGAN, Arjovsky et al., 2017).

**Variantes.**

- **DCGAN (2015):** deep convolutional GAN. Usa strided
  y transposed convs, BN, ReLU/LeakyReLU. Establece el
  baseline.
- **WGAN (2017):** Wasserstein loss con Lipschitz
  constraint (weight clipping o gradient penalty). Más
  estable, mejor gradiente.
- **LSGAN (2017):** least squares loss. Suaviza gradientes.
- **Conditional GAN (2014):** G y D reciben label y. Para
  generación condicional.
- **CycleGAN (2017):** traducción image-to-image sin
  pares. Para style transfer.

**Cuándo usar GANs.**

- Generación de imágenes: faces, art, textures.
- Super-resolución, inpainting, colorization.
- Augmentation de datos (con cuidado).
- Style transfer, image-to-image translation.

**Cuándo NO usar GANs.**

- Cuando se necesita training estable: usar VAE o
  diffusion models.
- Cuando se necesita mode coverage: GANs sufren mode
  collapse. Usar flows o autoregresivos.
- Para datos discretos: GANs son difíciles de entrenar
  en texto. Usar VAE con Gumbel-Softmax o
  transformers autoregresivos.

**Trampas.**

- **Mode collapse:** G produce 1-2 imágenes. Reducir
  learning rate de G, usar WGAN, o minibatch
  discrimination.
- **D perfecto:** si D accuracy ≈ 100%, G no aprende.
  Detener el entrenamiento de D cuando es demasiado
  bueno.
- **Sin normalización:** normalizar imágenes a [-1, 1]
  y usar Tanh en la salida de G. De lo contrario,
  exploding gradients.

## Constrúyelo

```python
import numpy as np


def generator_loss(D_fake):
    """Loss de G: queremos maximizar log D(G(z))."""
    return -np.log(D_fake + 1e-8).mean()


def discriminator_loss(D_real, D_fake):
    """Loss de D: queremos maximizar log D(x) + log(1 - D(G(z)))."""
    return -(np.log(D_real + 1e-8).mean()
            + np.log(1 - D_fake + 1e-8).mean())


def wgan_discriminator_loss(D_real, D_fake):
    """Wasserstein loss para D (critic).
    Maximizar D(real) - D(fake)."""
    return -(D_real.mean() - D_fake.mean())
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-gan
fase: 08
leccion: 03
---

Eres un asistente que ayuda con GANs. Recibirás la tarea
y los datos. Tu trabajo:

1. Para imágenes: DCGAN o StyleGAN. WGAN-GP para
   estabilidad.
2. Para mode collapse: unrolled GAN, minibatch
   discrimination, o WGAN.
3. Para datos pequeños (< 10k): data augmentation,
   transfer learning, o VAE.
4. Para texto: usar VAE con Gumbel-Softmax o
   transformers autoregresivos (no GANs).
5. Para style transfer: CycleGAN o pix2pix.
6. Para super-resolución: SRGAN, ESRGAN.
7. Advertir contra mode collapse y vanishing gradients.
8. Evaluar con FID o IS, no solo visual inspection.
```

## Ejercicios

1. **Mini-GAN**: implementa una GAN simple en
   MNIST.
2. **Mode collapse**: visualiza cuando G colapsa a
   pocas muestras.
3. **Desafío**: entrena StyleGAN2 en faces.

## Lecturas recomendadas

- *Generative Adversarial Nets* — Goodfellow et al., 2014.
- *DCGAN* — Radford et al., 2015.
- *Wasserstein GAN* — Arjovsky et al., 2017.
- *Improved Training of WGANs* — Gulrajani et al., 2017.

---

> 📚 **Adaptación al español** de la lección "[GANs: Generator and Discriminator]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
