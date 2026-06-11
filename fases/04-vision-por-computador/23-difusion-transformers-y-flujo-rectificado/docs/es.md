# Difusión transformers y flujo rectificado

> La evolucion mas alla de DDPM: DiT (transformer en vez de U-Net) y rectified flow (trayectorias lineales). Base de SD3, FLUX, Sora. Sampling en 1-4 pasos con distillation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-stable-diffusion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Comparar DDPM (beta schedule) vs rectified flow (lineal).
- Implementar interpolacion lineal x_t = (1-t) x_0 + t * eps.
- Entrenar red para predecir velocidad v = eps - x0.
- Diagnosticar distillation, consistency models.

## Constrúyelo

```python
def interpolacion_rectified(x0, ruido, t):
    return (1 - t) * x0 + t * ruido


def velocidad_target(x0, ruido):
    return ruido - x0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-diffusion-2024
fase: 04
leccion: 23
---

1. SOTA: FLUX.1-dev, SD3.
2. Rapido: SD3-Turbo, FLUX.1-schnell.
3. Custom: FLUX + LoRA.
4. Video: Wan2.1, HunyuanVideo.
5. DiT > U-Net, flow matching > DDPM.
```

## Ejercicios

1. **Flow matching completo**: implementar ODE solver (Euler
   o RK45) para sampling.
2. **Consistency model**: implementar la loss de consistencia
   (mismo x_0 desde cualquier t).
3. **Desafio**: fine-tunear FLUX.1-dev con LoRA en 20
   imagenes y reportar FID.

## Lecturas recomendadas

- "DiT" (Peebles & Xie, 2023)
- "Flow Matching" (Lipman et al., 2023)
- "Consistency Models" (Song et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Diffusion Transformers and Rectified Flow]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).