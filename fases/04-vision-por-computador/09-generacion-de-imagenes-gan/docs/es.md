# Generación de imágenes con GANs

> La invencion que aprendio a generar imagenes realistas sin probabilidad explicita. Dos redes compiten: el generador intenta enganar, el discriminador intenta no dejarse enganar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar generador y discriminador minimos.
- Calcular BCE sobre logits.
- Entender la dinamica adversarial.

## Constrúyelo

```python
def generador_minimo(z, salida_dim=784):
    h = leaky_relu(z @ W1)
    out = np.tanh(h @ W2)
    return out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-gan-vs-diffusion
fase: 04
leccion: 09
---

1. Tiempo real: GAN (StyleGAN, BigGAN).
2. Max calidad: Diffusion.
3. Distribucional: VAE.
4. Adam lr=2e-4, beta1=0.5.
5. FID para comparar.
```

## Ejercicios

1. **Wasserstein loss**: implementa WGAN con weight
   clipping.
2. **Spectral normalization**: normaliza la matriz de pesos
   por su mayor valor singular.
3. **Desafio**: entrenar DCGAN en CIFAR-10 y reportar FID.

## Lecturas recomendadas

- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Wasserstein GAN" (Arjovsky et al., 2017)
- "Spectral Normalization for GANs" (Miyato et al., 2018)

---

> 📚 **Adaptación al español** de la lección "[Image Generation with GANs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).