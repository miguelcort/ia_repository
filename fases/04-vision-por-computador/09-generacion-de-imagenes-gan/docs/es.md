# 09 — Generación de imágenes: GAN

> Las GANs (Generative Adversarial Networks) enfrentan dos redes: una que genera imágenes falsas y otra que intenta distinguirlas de las reales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la arquitectura DCGAN desde cero.
- Entender el equilibrio Nash entre generador y
  discriminador.
- Diagnosticar mode collapse, vanishing generator y
  oscilación.
- Conocer pérdidas alternativas: LSGAN, WGAN, hinge.

## El problema

Las GANs (Goodfellow et al., 2014) son el primer modelo
generativo que producía imágenes fotorrealistas a alta
resolución. La idea es simple y poderosa: dos redes
compiten. El generador (G) intenta producir imágenes que
parezcan reales; el discriminador (D) intenta distinguir
las reales de las falsas. Ambos mejoran en el proceso.
Diez años después, las GANs han sido superadas por
difusión en calidad, pero siguen siendo útiles para
generación condicional, super-resolución y translation.

## El concepto

**Objetivo adversarial.** Min-max:

```text
min_G max_D E[log D(x_real)] + E[log(1 - D(G(z)))]
```

donde `z` es ruido aleatorio y `G(z)` es la imagen
generada. D quiere maximizar la probabilidad de
discriminar; G quiere minimizar la probabilidad de que
D acierte.

**DCGAN (Radford et al., 2015).** La primera GAN con
convoluciones. Generador: toma `z ∈ R^100`, lo proyecta a
un feature map pequeño con `ConvTranspose2D`, luego upsamples
progresivamente hasta la imagen final. Discriminador: CNN
clásica que emite un escalar (real o fake).

**Equilibrio Nash.** En teoría, el entrenamiento converge
a un equilibrio donde G produce la distribución real y D no
puede distinguir (`D(x) = 0.5` en todas partes). En la
práctica, este equilibrio es inestable: o D se vuelve
perfecto y G no recibe gradiente (vanishing G), o G colapsa
a un solo modo (mode collapse).

**Mode collapse.** G produce solo unas pocas imágenes
distintas, ignorando la diversidad del training set.
Síntoma: D loss baja, G loss también, pero las muestras
son casi idénticas.

**Vanishing G.** Si D se vuelve demasiado fuerte, su
gradiente sobre las muestras de G se satura a 0. G deja de
aprender.

**Pérdidas alternativas para estabilizar.**

- **LSGAN (Least Squares):** reemplaza BCE con MSE.
  Gradientes más suaves.
- **WGAN (Wasserstein):** usa la distancia
  Wasserstein-1 entre distribuciones. Requiere D
  (llamado "crítico") 1-Lipschitz (vía weight clipping
  o gradient penalty).
- **WGAN-GP:** gradient penalty en lugar de weight
  clipping. Más estable.
- **Hinge loss:** variante que satura menos.
- **Spectral normalization:** normaliza los pesos del
  discriminador por su mayor valor singular. Estabiliza
  el entrenamiento sin penalización explícita.

**Técnicas adicionales.**

- **Minibatch discrimination:** D mira batches completos
  para detectar mode collapse.
- **Feature matching:** loss para G que iguala features
  intermedios de D en imágenes reales y falsas.
- **Label smoothing:** D usa 0.9 en vez de 1.0 para
  imágenes reales. Reduce confianza de D.
- **TTUR (Two Time-Scale Update Rule):** lr de D mayor
  que lr de G.

**Métricas de evaluación.**

- **Inception Score (IS):** mide la calidad y diversidad
  de las muestras usando un Inception preentrenado.
- **FID (Fréchet Inception Distance):** compara la
  distribución de features de Inception entre reales y
  generadas. Más robusto que IS.
- **KID (Kernel Inception Distance):** variante
  unbiased de FID para datasets pequeños.

**Trampas.**

- **Mode collapse silencioso:** pérdida de G no captura la
  falta de diversidad. Visualizar muestras regularmente.
- **D demasiado fuerte:** G no aprende. Usar spectral
  normalization o reducir capacidad de D.
- **D demasiado débil:** G no se ve forzado a mejorar.
  Aumentar capacidad de D.
- **Learning rate fijo sin decay:** el equilibrio Nash es
  dinámico. Usar lr decay ayuda a estabilizar.

## Constrúyelo

```python
import numpy as np


def dcgan_generator(z, W_deconv, b_deconv):
    """Generador DCGAN simplificado. z: (batch, 100).
    Devuelve (batch, H, W, 3) en [-1, 1]."""
    h = z
    for w, b in zip(W_deconv, b_deconv):
        h = np.dot(h, w.T) + b
        h = np.maximum(0.2 * h, h)  # LeakyReLU
    h = np.tanh(h)
    return h


def dcgan_discriminator(x, W_conv, b_conv):
    """Discriminador DCGAN. x: (batch, H, W, 3). Devuelve
    logits real/fake."""
    h = x
    for w, b in zip(W_conv, b_conv):
        h = np.dot(h, w) + b
        h = np.maximum(0.2 * h, h)  # LeakyReLU
    return h.mean(axis=(1, 2, 3))  # global avg pool


def g_loss(d_fake):
    """Pérdida del generador (no saturante)."""
    return float(np.mean(-np.log(np.maximum(d_fake, 1e-12))))


def d_loss(d_real, d_fake):
    """Pérdida del discriminador."""
    return float(
        -np.mean(np.log(np.maximum(d_real, 1e-12)))
        - np.mean(np.log(np.maximum(1 - d_fake, 1e-12)))
    )


def wasserstein_d_loss(d_real, d_fake):
    return float(np.mean(d_fake) - np.mean(d_real))
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
fase: 04
leccion: 09
---

Eres un asistente que ayuda a entrenar una GAN. Recibirás
el dataset, el hardware y la calidad objetivo. Tu trabajo:

1. Empezar con DCGAN como baseline.
2. Si tienes suficiente cómputo: StyleGAN3 para alta
   resolución.
3. Para estabilizar: spectral normalization en D, o
   WGAN-GP.
4. Si mode collapse: minibatch discrimination o
   unrolling.
5. Optimizer: Adam con lr=2e-4, beta1=0.5 (D y G con
   lr similares).
6. Monitorear D loss y G loss: deben oscilar, no
   converger a 0.
7. Evaluar con FID cada N epochs.
8. Visualizar muestras fijas en TensorBoard
   para detectar mode collapse.
```

## Ejercicios

1. **DCGAN mínimo**: implementa DCGAN y entrena en
   MNIST.
2. **Mode collapse**: detecta mode collapse midiendo la
   diversidad de muestras.
3. **Desafío**: implementa WGAN-GP con gradient penalty
   y compara estabilidad con DCGAN.

## Lecturas recomendadas

- *Generative Adversarial Nets* — Goodfellow et al., 2014.
- *Unsupervised Representation Learning with DCGAN* —
  Radford et al., 2015.
- *Wasserstein GAN* — Arjovsky et al., 2017.
- *Spectral Normalization for GANs* — Miyato et al., 2018.
- pytorch-fid: <https://github.com/mseitzer/pytorch-fid>.

---

> 📚 **Adaptación al español** de la lección "[Image Generation: GAN]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
