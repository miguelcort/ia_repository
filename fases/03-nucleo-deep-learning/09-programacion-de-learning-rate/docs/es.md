# Programación del learning rate

> El learning rate no deberia ser constante. Subir al inicio (warmup), bajar al final (decay). La forma del decay importa mas de lo que parece.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-optimizadores
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar 5 schedules: constante, step, exponential, cosine, warmup+cosine.
- Implementar reduce-on-plateau reactivo.
- Diagnosticar cuando cada uno aplica.

## Constrúyelo

```python
def lr_cosine(paso, lr_max=0.1, lr_min=0.0, T_max=100):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * paso / T_max))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-lr-schedule
fase: 03
leccion: 09
---

1. Vision: cosine annealing, lr=0.1/1e-3.
2. NLP: warmup 10% + cosine, lr=1e-4.
3. RL: constante o linear.
4. Plateau: reduce-on-plateau factor 0.5.
5. Fine-tuning: lr bajo + cosine corto.
```

## Ejercicios

1. **Cosine warm restarts**: cosine + reinicio a lr_max cada
   T_max pasos.
2. **OneCycle**: lr sube y luego baja en una sola epoca.
3. **Desafio**: implementa un trainer con AdamW + cosine +
   warmup + gradient clipping y entrena un MLP en MNIST.

## Lecturas recomendadas

- "SGDR: Stochastic Gradient Descent with Warm Restarts"
  (Loshchilov & Hutter, 2016)
- "Super-Convergence" (Smith & Topin, 2018) — OneCycle
- "Attention is All You Need" (Vaswani et al., 2017) — warmup

---

> 📚 **Adaptación al español** de la lección "[Learning Rate Schedules]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).