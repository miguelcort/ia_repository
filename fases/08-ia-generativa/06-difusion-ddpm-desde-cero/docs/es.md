# Difusión DDPM desde cero

> DDPM (Ho 2020): forward q(x_t|x_0) añade ruido gaussiano en T steps. Reverse p_θ(x_{t-1}|x_t) aprende a denoising. Loss: ‖ε - ε_θ(x_t, t)‖². Schedule: linear o cosine (Improved DDPM). U-Net predice ruido. Variantes: DDIM (50-100 steps), latent diffusion (VAE + diffusion en latente), flow matching, DiT. SOTA en image generation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/03-gans-generador-y-discriminador
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar linear y cosine beta schedule.
- Implementar forward process q(x_t|x_0).
- Implementar reverse mean y variance.
- Diagnosticar el loss MSE.

## Constrúyelo

```python
def q_sample(x0, t, alpha_bar, noise=None, seed=0):
    a = alpha_bar[t]
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * noise, noise
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ddpm
fase: 08
leccion: 06
---

1. Forward: x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps.
2. Loss: ||eps - eps_theta(x_t, t)||^2.
3. Schedule: linear o cosine.
4. U-Net o DiT.
5. Variantes: DDIM, latent, flow matching.
```

## Ejercicios

1. **DDPM**: entrenar U-Net pequeno en MNIST
   con diffusion.
2. **DDIM**: implementar sampling en 50 steps.
3. **Desafio**: implementar classifier-free
   guidance.

## Lecturas recomendadas

- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
- "Denoising Diffusion Implicit Models" (Song et al., 2020)
- "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021)

---

> 📚 **Adaptación al español** de la lección "[Diffusion DDPM From Scratch]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).