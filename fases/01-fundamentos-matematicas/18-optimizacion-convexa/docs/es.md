# Optimizacion convexa

> Si tu problema es convexo, todo es mas facil. Saber si lo es vale oro.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-optimizacion-familia-gradiente
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Verificar convexidad via la Hessiana.
- Aplicar gradiente descendente a funciones convexas.
- Diagnosticar por que deep learning no es convexo.

## Constrúyelo

```python
import numpy as np


def es_convexa_2d(f, x, h=1e-5):
    fxx = (f(x[0] + h, x[1]) - 2 * f(x[0], x[1]) + f(x[0] - h, x[1])) / h**2
    fyy = (f(x[0], x[1] + h) - 2 * f(x[0], x[1]) + f(x[0], x[1] - h)) / h**2
    fxy = (f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h) - f(x[0] - h, x[1] + h) + f(x[0] - h, x[1] - h)) / (4 * h**2)
    hess = np.array([[fxx, fxy], [fxy, fyy]])
    autovalores = np.linalg.eigvalsh(hess)
    return bool(np.all(autovalores >= -1e-6))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-convex-check
fase: 01
leccion: 18
---

1. Verificar convexidad (Hessiana PSD).
2. Si convexa, gradiente conjugado/LBFGS.
3. Si no, advertir minimo local.
4. Para deep learning, lr schedule + multiple seeds.
```

## Ejercicios

1. **CVXPY**: instala cvxpy y resuelve un LP simple.
2. **KKT**: implementa las condiciones KKT para un problema con
   restricciones.
3. **Desafio**: implementa gradiente conjugado lineal.

## Lecturas recomendadas

- "Convex Optimization" (Boyd & Vandenberghe): <https://stanford.edu/~boyd/cvxbook/>
- CVXPY: <https://www.cvxpy.org/>

---

> 📚 **Adaptación al español** de la lección "[Convex Optimization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).