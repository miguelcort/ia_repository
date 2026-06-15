# 18 — Optimización convexa

> Si tu problema es convexo, todo es más fácil. Saber si lo es vale oro.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-optimizacion-familia-gradiente
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Verificar convexidad de una función vía su Hessiana.
- Aplicar gradiente descendente a funciones convexas y
  diagnosticar convergencia.
- Diagnosticar por qué deep learning *no* es convexo y qué
  implicaciones tiene.
- Conocer las condiciones KKT y cuándo usarlas.

## El problema

Hay dos tipos de problemas de optimización: los convexos, donde
*todo* mínimo local es global, y los no convexos, donde el
gradiente puede estancarse en un punto silla o un mínimo
local. Saber a cuál pertenece tu problema cambia
radicalmente la estrategia: en convexo, gradiente descendente
converge siempre; en no convexo, necesitas LR schedules,
múltiples seeds, y cuidado con los puntos silla. La lección
cubre el test de convexidad vía la Hessiana y las herramientas
de referencia.

## El concepto

**Definición.** Una función `f: Rⁿ → R` es **convexa** si para
todo `x, y` y `t ∈ [0, 1]`:

```text
f(tx + (1-t)y) ≤ t f(x) + (1-t) f(y)
```

Equivalente: el segmento entre dos puntos de la gráfica queda
por encima o sobre la gráfica. Implicación clave: *cualquier*
mínimo local es global.

**Test de la Hessiana.** Para `f` dos veces diferenciable, `f`
es convexa si y solo si su Hessiana `H(x) = ∇²f(x)` es
**semidefinida positiva** (PSD) en todo `x`:
`yᵀ H(x) y ≥ 0` para todo `y`. En la práctica, calculas los
autovalores y verificas que todos sean `≥ 0` (con tolerancia
numérica).

**Funciones convexas comunes.** MSE, cross-entropy, norma L2,
hinge loss, logistic loss. **No convexas:** la composición
*max* de convexas, las redes neuronales profundas, cualquier
función con un máximo local.

**Gradiente descendente en convexo.** Convergencia `O(1/k)`
para funciones convexas suaves (L-smooth) con paso fijo, o
`O(1/k²)` con Nesterov. Tasas mucho mejores que en no convexo,
donde solo puedes decir "converge a un crítico".

**KKT (Karush-Kuhn-Tucker).** Generalización de Lagrange para
problemas con restricciones de desigualdad. Si el problema es
convexo y satisface una condición de regularidad (Slater), KKT
es necesario y suficiente para optimalidad. La lección
introduce el sistema; CVXPY resuelve KKT internamente.

**Por qué deep learning no es convexo.** Una red neuronal
componiendo activaciones ReLU produce una función objetivo
**no convexa** en los pesos. La Hessiana tiene autovalores
positivos y negativos. La optimización usa SGD con LR schedule
y variantes como Adam, pero no puedes garantizar convergencia a
un óptimo global. La práctica funciona porque:

1. Los puntos silla dominan al principio; SGD los evade
   fácilmente.
2. Los mínimos locales de redes sobre-parametrizadas tienen
   loss cercano al global.
3. El ruido del mini-batch añade regularización implícita.

**Métodos prácticos para convexo.**

| Problema | Método |
|---|---|
| Smooth convex, pequeña escala | L-BFGS |
| Smooth convex, grande escala | SGD con LR schedule |
| Strongly convex | Newton o gradiente conjugado |
| Con restricciones lineales | Projected gradient o KKT |
| Convex no diferenciable | Subgradiente, proximal |

## Constrúyelo

```python
import numpy as np


def es_convexa_2d(f, x, h=1e-5):
    """Test de convexidad por Hessiana numérica."""
    fxx = (f(x[0] + h, x[1]) - 2 * f(x[0], x[1]) + f(x[0] - h, x[1])) / h**2
    fyy = (f(x[0], x[1] + h) - 2 * f(x[0], x[1]) + f(x[0], x[1] - h)) / h**2
    fxy = (
        f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h)
        - f(x[0] - h, x[1] + h) + f(x[0] - h, x[1] - h)
    ) / (4 * h**2)
    hess = np.array([[fxx, fxy], [fxy, fyy]])
    autovalores = np.linalg.eigvalsh(hess)
    return bool(np.all(autovalores >= -1e-6))


def gradiente_descendente(f, grad_f, x0, lr=0.1, n_iter=100,
                         tol=1e-6):
    """Para funciones convexas L-smooth."""
    x = np.asarray(x0, dtype=float)
    for i in range(n_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x, i
        x = x - lr * g
    return x, n_iter
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

Eres un asistente que ayuda a diagnosticar la convexidad de un
problema de optimización. Recibirás la función objetivo y
restricciones. Tu trabajo:

1. Verificar convexidad (Hessiana PSD).
2. Si convexa, recomendar L-BFGS o gradiente conjugado.
3. Si no convexa, advertir sobre mínimos locales.
4. Para deep learning, recomendar LR schedule + múltiples
   seeds + monitorizar gradientes.
5. Sugerir CVXPY si el problema es convexo y de tamaño
   moderado.
```

## Ejercicios

1. **CVXPY**: instala cvxpy y resuelve un LP simple con
   restricciones lineales.
2. **KKT**: implementa las condiciones KKT para un problema
   con restricciones.
3. **Desafío**: implementa gradiente conjugado lineal y
   compáralo con gradiente descendente en una cuadrática
   mal condicionada.

## Lecturas recomendadas

- *Convex Optimization* — Boyd & Vandenberghe: <https://stanford.edu/~boyd/cvxbook/>.
- CVXPY: <https://www.cvxpy.org/>.
- *Optimization Models* — Calafiore & El Ghaoui.

---

> 📚 **Adaptación al español** de la lección "[Convex Optimization]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
