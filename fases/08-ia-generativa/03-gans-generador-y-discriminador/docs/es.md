# GANs generador y discriminador

> GAN (Goodfellow 2014): G(z) genera fake, D(x) clasifica real/fake, entrenamiento adversarial. Problemas: mode collapse, vanishing gradient, training inestable. Soluciones: WGAN (Wasserstein, gradient penalty), spectral norm, two-time-scale. DCGAN: ConvTranspose + batch norm + leaky ReLU. StyleGAN3 SOTA para caras. Híbridos: VAE-GAN, latent diffusion.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/02-autoencoders-y-vae
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Generator y Discriminator.
- Calcular pérdidas GAN (vanilla, non-saturating).
- Implementar Wasserstein loss.
- Diagnosticar mode collapse y vanishing gradient.

## Constrúyelo

```python
def generator_forward(z, W1, b1, W2, b2):
    h = np.maximum(0, z @ W1 + b1)
    return sigmoid(h @ W2 + b2)
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

1. GAN: G vs D, adversarial.
2. D_loss = -log D(x) - log(1-D(G(z))).
3. WGAN: Wasserstein, 1-Lipschitz.
4. Mode collapse, vanishing gradient.
5. StyleGAN, BigGAN, WGAN-GP.
```

## Ejercicios

1. **Entrenar DCGAN** en MNIST o CelebA.
   Medir FID e inception score.
2. **WGAN-GP**: implementar gradient penalty
   y comparar estabilidad.
3. **Desafio**: implementar conditional GAN
   (CGAN) para clase condicional.

## Lecturas recomendadas

- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Wasserstein GAN" (Arjovsky et al., 2017)
- "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017)
- "Unsupervised Representation Learning with Deep Convolutional GANs" (Radford et al., 2015)

---

> 📚 **Adaptación al español** de la lección "[GANs Generator Discriminator]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).