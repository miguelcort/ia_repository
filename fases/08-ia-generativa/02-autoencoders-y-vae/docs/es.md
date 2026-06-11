# Autoencoders y VAE

> VAE: encoder → (μ, logvar) → reparameterization z = μ + σ·ε → decoder → x_recon. Loss: ELBO = reconstruction + KL(q(z|x) ‖ p(z)). Reparameterization trick habilita backprop. KL collapse: annealing, free bits, β-VAE. VQ-VAE: latente discreto con codebook. Aplicaciones: anomaly detection, generation, denoising, disentanglement.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar reparameterization trick.
- Calcular KL divergence.
- Implementar VAE forward completo.
- Diagnosticar KL collapse.

## Constrúyelo

```python
def reparameterize(mu, logvar, seed=0):
    eps = rng.standard_normal(mu.shape)
    return mu + np.exp(0.5 * logvar) * eps

def kl_divergence(mu, logvar):
    return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vae
fase: 08
leccion: 02
---

1. VAE: encoder -> (mu, logvar) -> z -> decoder.
2. Loss: ELBO = recon + KL.
3. Reparameterization: z = mu + sigma*eps.
4. KL collapse: annealing, free bits, beta-VAE.
5. VQ-VAE: codebook discreto.
```

## Ejercicios

1. **Anomaly detection**: entrenar VAE en MNIST
   y detectar anomalias con recon error.
2. **beta-VAE**: implementar disentanglement
   en CelebA.
3. **Desafio**: implementar VQ-VAE y
   entrenar AR sobre los latentes.

## Lecturas recomendadas

- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "Neural Discrete Representation Learning" (van den Oord et al., 2017)
- "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (Higgins et al., 2017)

---

> 📚 **Adaptación al español** de la lección "[Autoencoders VAE]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).