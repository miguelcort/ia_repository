# Generación de imágenes con difusión

> La idea contraintuitiva: aprender a generar imagenes aprendiendo a quitar ruido. Sorprendentemente simple (MSE sobre el ruido) y absurdamente efectivo. La base de Stable Diffusion, DALL-E 3, Midjourney.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-generacion-de-imagenes-gan
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar beta schedule lineal.
- Implementar forward diffusion.
- Entender DDIM y el sampling.
- Diagnosticar CFG y conditioning.

## Constrúyelo

```python
def forward_diffusion(x0, t, alpha_bars, semilla=0):
    ruido = np.random.default_rng(semilla).normal(0, 1, size=x0.shape)
    return np.sqrt(alpha_bars[t]) * x0 + np.sqrt(1 - alpha_bars[t]) * ruido, ruido
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-diffusion-elegir
fase: 04
leccion: 10
---

1. Produccion: SDXL, DALL-E 3, Imagen.
2. Custom: SDXL + LoRA / DreamBooth.
3. Rapido: SD-Turbo (1-4 pasos).
4. DDIM 20-50 pasos, CFG w=7-12.
5. diffusers + accelerate.
```

## Ejercicios

1. **DPM++ sampler**: implementa multi-step ODE solver.
2. **Classifier-free guidance**: entrenar con 10% uncond y
   mezclar en inference.
3. **Desafio**: fine-tunear SDXL con LoRA en 10 imagenes
   de un concepto custom.

## Lecturas recomendadas

- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Denoising Diffusion Implicit Models" (Song et al., 2020)
- diffusers: <https://huggingface.co/docs/diffusers/>

---

> 📚 **Adaptación al español** de la lección "[Image Generation with Diffusion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).