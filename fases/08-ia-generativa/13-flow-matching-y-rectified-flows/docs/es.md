# Flow matching y rectified flows

> Flow matching (Lipman 2022): aprende ODE dx_t/dt = v_θ(x_t, t) entre distribuciones. Loss: ‖v - v_θ‖². Rectified flow: linear interpolation x_t = t·x_1 + (1-t)·x_0, paths straight, optimal transport. Inference: Euler, RK4, Heun, 10-50 steps vs 1000 DDPM. SD3 (Esser 2024), FLUX (Black Forest Labs), Stable Cascade usan flow matching. 10-100x speedup vs DDPM.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/06-difusion-ddpm-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar interpolación lineal.
- Calcular velocity y flow matching loss.
- Implementar ODE solver (Euler).
- Diagnosticar diffusion vs flow matching.

## Constrúyelo

```python
def conditional_flow(x0, x1, t):
    return (1 - t) * x0 + t * x1, x1 - x0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-flow-matching
fase: 08
leccion: 13
---

1. ODE: dx/dt = v_theta(x_t, t).
2. Loss: ||v - v_theta||^2.
3. Rectified flow: linear paths.
4. Heun/Euler/RK4 10-50 steps.
5. SD3, FLUX SOTA.
```

## Ejercicios

1. **Flow matching**: entrenar modelo
   pequeno con flow matching.
2. **Rectified flow**: implementar y
   comparar paths.
3. **Desafio**: implementar distillation
   1-step consistency model.

## Lecturas recomendadas

- "Flow Matching for Generative Modeling" (Lipman et al., 2022)
- "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (Liu et al., 2022)
- "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser et al., 2024)
- "Flow Matching Guide" (Meta AI)

---

> 📚 **Adaptación al español** de la lección "[Flow Matching Rectified Flows]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).