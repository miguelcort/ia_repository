# Introducción a JAX

> numpy + autodiff + compilación XLA + paralelismo multi-device. JAX brilla donde TPU y computo numérico de alto rendimiento se encuentran. PyTorch brilla donde el prototipado y el debug rápido importan.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-introduccion-a-pytorch
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Conocer la API de JAX: jax.numpy, grad, jit, vmap, pmap.
- Implementar autodiff con grad.
- Diagnosticar cuando JAX gana a PyTorch (TPU, HPC).

## Constrúyelo

```python
import jax.numpy as jnp
from jax import grad, jit, vmap

def f(x):
    return jnp.sum(x ** 2)

g = grad(f)         # gradiente automatico
f_jit = jit(f)      # compilacion XLA
f_vmapped = vmap(f) # vectorizacion sobre batch
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-jax-vs-pytorch
fase: 03
leccion: 12
---

1. TPU / escala: JAX + Flax + Optax.
2. Prototipado: PyTorch.
3. Numerico: JAX (jit, vmap).
4. Reglas: funciones puras, jit primera vez lento.
```

## Ejercicios

1. **Custom grad**: implementa @jax.custom_vjp para una
   operacion no diferenciable.
2. **PMAP multi-GPU**: paraleliza un loop con pmap.
3. **Desafio**: reimplementa el mini-framework L10 en JAX
   con grad + jit.

## Lecturas recomendadas

- JAX docs: <https://jax.readthedocs.io/>
- "JAX: A Framework for HPC and ML" (Bradbury et al.)
- Flax: <https://flax.readthedocs.io/>

---

> 📚 **Adaptación al español** de la lección "[Introduction to JAX]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).