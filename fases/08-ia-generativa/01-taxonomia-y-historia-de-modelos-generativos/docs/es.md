# Taxonomía e historia de modelos generativos

> Familias: VAE (ELBO, estable, blurry), GAN (adversarial, sharp, mode collapse), AR (likelihood exacto), Normalizing flow (biyectivo), Diffusion (denoising, SOTA, lento), Flow matching (ODE, rápido). Hitos: VAE 2013, GAN 2014, DDPM 2020, Stable Diffusion 2022, DiT/Sora 2024. Métricas: FID, IS, CLIP score, precision/recall, human eval.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Comparar familias generativas (VAE, GAN, AR, flow, diffusion, flow matching).
- Diagnosticar trade-offs (calidad, likelihood, velocidad, estabilidad).
- Conocer historia y hitos.
- Seleccionar métricas de evaluación.

## Constrúyelo

```python
def taxonomy_summary():
    return [
        ("VAE", "Encoder-decoder probabilistico, ELBO, blurry"),
        ("GAN", "Adversarial, sharp, mode collapse"),
        ("AR", "Autoregresivo, transformer, likelihood exacto"),
        ("Normalizing flow", "Biyective, likelihood exacto, denso"),
        ("Diffusion", "Denoising, SOTA imagen, lento"),
        ("Flow matching", "ODE path, 10-50 steps, rapido"),
    ]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-generative-taxonomy
fase: 08
leccion: 01
---

1. VAE, GAN, AR, Flow, Diffusion, Flow matching.
2. Trade-offs: quality, likelihood, mode, speed.
3. FID, IS, CLIP score para evaluar.
4. Hibridos: VAE-GAN, latent diffusion, DiT.
5. Hoy: diffusion imagen, AR texto, flow matching.
```

## Ejercicios

1. **Comparar**: entrenar VAE, GAN, y diffusion
   pequeno en MNIST. Comparar FID.
2. **Hitos**: investigar otro modelo (e.g.
   NeRF, 3D diffusion) y agregar a taxonomia.
3. **Desafio**: implementar metricas de
   evaluacion generativa.

## Lecturas recomendadas

- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Flow Matching for Generative Modeling" (Lipman et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Generative Models Taxonomy History]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).