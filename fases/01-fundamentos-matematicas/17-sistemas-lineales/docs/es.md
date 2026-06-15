# 17 — Sistemas lineales

> Casi todo modelo lineal (regresión, KKT, sistemas de ecuaciones) termina resolviendo Ax = b.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-descomposicion-svd
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Resolver `Ax = b` con `np.linalg.solve` y entender cuándo es
  estable.
- Aplicar mínimos cuadrados con `np.linalg.lstsq` cuando `A` es
  rectangular o está mal condicionada.
- Diagnosticar condicionamiento numérico con el número de
  condición `kappa(A)`.
- Conocer factorización LU, Cholesky y QR como herramientas
  alternativas.

## El problema

Resolviste el sistema y obtuviste `x = (1e9, -1e9, 1e-3)`. ¿Es
ese el resultado correcto o el sistema está mal condicionado?
La lección cubre la teoría mínima indispensable para que el
estudiante detecte problemas numéricos antes de publicar
resultados que parecen precisos pero están dominados por
*roundoff error*.

## El concepto

**Sistemas cuadrados bien condicionados.** Para `A` cuadrada e
invertible, `np.linalg.solve(A, b)` usa factorización LU bajo el
capó: descompone `A = LU` y resuelve `Ly = b`, `Ux = y`. Es `O(n³)`
y numéricamente estable para matrices bien condicionadas.

**Sistemas rectangulares o singulares.** Cuando `A` es
rectangular (más ecuaciones que incógnitas, o al revés) o
cuadrada pero singular, `solve` falla. La solución general es
mínimos cuadrados: encontrar `x` que minimiza `||Ax - b||²`. La
función `np.linalg.lstsq(A, b)` devuelve la solución de norma
mínima, residuos, rango y valores singulares.

**Número de condición `kappa(A)`.** Mide cuánto amplifica `A`
los errores de input. `kappa = ||A|| * ||A⁻¹||` (en alguna
norma). Si `kappa = 1e10`, la matriz está mal condicionada y
pequeños cambios en `b` producen cambios enormes en `x`. La
regla práctica: `kappa > 1/eps_máquina ≈ 1e16` indica que la
solución no es confiable.

```python
import numpy as np
kappa = np.linalg.cond(A)
if kappa > 1e10:
    print("mal condicionado, considera regularización")
```

**Factorización LU.** `A = LU` donde `L` es triangular inferior
y `U` es triangular superior. Útil para resolver el mismo
sistema con muchos `b` distintos. NumPy la usa por dentro en
`solve`.

**Factorización de Cholesky.** Para `A` simétrica definida
positiva (SDP), `A = LLᵀ` donde `L` es triangular inferior. Es
2x más rápida que LU y numéricamente más estable. Si falla,
sabes que `A` no es SDP.

**Factorización QR.** `A = QR` donde `Q` es ortogonal y `R` es
triangular superior. Es la base de mínimos cuadrados
numéricamente estables: `Ax = b` se resuelve como `Rx = Qᵀb`.

**Cuándo usar cada método.**

| Situación | Método |
|---|---|
| `A` cuadrada invertible bien condicionada | `solve` (LU) |
| `A` cuadrada, mal condicionada, invertible | `lstsq` o Cholesky si SDP |
| `A` rectangular (más ecuaciones que inc.) | `lstsq` (mínimos cuadrados) |
| `A` rectangular (más inc. que ecuaciones) | `lstsq` (norma mínima) |
| `A` SDP y bien condicionada | Cholesky |
| Sistema sparse grande | `scipy.sparse.linalg.spsolve` |
| Sistema con muchos `b` distintos | LU o Cholesky precomputada |

## Constrúyelo

```python
import numpy as np


def resolver(A, b):
    """Resuelve Ax = b. Asume A cuadrada e invertible."""
    return np.linalg.solve(A, b)


def lstsq(A, b):
    """Mínimos cuadrados para A rectangular o mal condicionada."""
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return x


def condicion(A):
    """Número de condición. >1e10 es mal condicionada."""
    return float(np.linalg.cond(A))


def cholesky(A):
    """A = L L^T. Solo para A simétrica definida positiva."""
    return np.linalg.cholesky(A)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-linear-system
fase: 01
leccion: 17
---

Eres un asistente que ayuda a resolver sistemas lineales.
Recibirás una descripción (forma de A, número de ecuaciones,
rango, condicionamiento) y debes:

1. Si A es cuadrada e invertible, usar `solve`.
2. Si A es singular, mal condicionada o rectangular,
   `lstsq` o Cholesky si es SDP.
3. Si la solución es muy grande, sospechar mal condicionamiento.
4. Para sistemas sparse, `scipy.sparse.linalg`.
5. Para no cuadrados, `lstsq`.
6. Reportar el número de condición antes de aceptar la
   solución.
```

## Ejercicios

1. **Regresión lineal múltiple**: con 5 features, ajusta el
   modelo y verifica residuos.
2. **Ridge regression**: implementa con `lstsq` +
   regularización `λI`.
3. **Desafío**: implementa KKT para SVM lineal (formulación
   primal-dual).

## Lecturas recomendadas

- *Numerical Linear Algebra* — Trefethen & Bau.
- LAPACK: <https://netlib.org/lapack/>.
- *Matrix Computations* — Golub & Van Loan.

---

> 📚 **Adaptación al español** de la lección "[Linear Systems]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
