# Conditional GANs y pix2pix

> Conditional GAN: G(z, c) y D(x, c) condicionados en c (label, image, text). Variantes: cGAN concat, projection discriminator (Miyato 2018). pix2pix: image-to-image supervisada, U-Net + PatchGAN, loss L1 + cGAN. CycleGAN: image-to-image sin pares, cycle + identity + adversarial. Hoy: text-to-image con diffusion + CLIP (Stable Diffusion, Imagen, DALL-E 3).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/03-gans-generador-y-discriminador
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar G(z, c) y D(x, c) condicionales.
- Implementar projection discriminator.
- Calcular L1 + adversarial loss (pix2pix).
- Diagnosticar PatchGAN vs full-image D.

## Constrúyelo

```python
def conditional_generator(z, c, W1, b1, W2, b2):
    h = np.concatenate([z, c], axis=-1)
    h = np.maximum(0, h @ W1 + b1)
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
name: prompt-cgan-pix2pix
fase: 08
leccion: 04
---

1. CGAN: G(z, c), D(x, c).
2. Projection D: estable.
3. pix2pix: L1 + cGAN.
4. PatchGAN: high-freq texture.
5. CycleGAN: cycle + identity.
```

## Ejercicios

1. **pix2pix**: entrenar en edges2shoes
   dataset.
2. **CycleGAN**: implementar y entrenar en
   horses2zebra.
3. **Desafio**: implementar conditional
   diffusion con classifier-free guidance.

## Lecturas recomendadas

- "Conditional Generative Adversarial Nets" (Mirza & Osindero, 2014)
- "Image-to-Image Translation with Conditional Adversarial Networks" (Isola et al., 2017)
- "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (Zhu et al., 2017)
- "Projection Discriminator" (Miyato & Koyama, 2018)

---

> 📚 **Adaptación al español** de la lección "[Conditional GANs Pix2Pix]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).