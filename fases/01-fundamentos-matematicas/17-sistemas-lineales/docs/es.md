# Sistemas lineales

> Casi todo modelo lineal (regresion, KKT, sistemas de ecuaciones) termina resolviendo Ax = b.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-descomposicion-svd
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Resolver Ax = b con np.linalg.solve.
- Aplicar minimos cuadrados con np.linalg.lstsq.
- Diagnosticar condicion numerico.
- Conocer factorizacion LU y SVD.

## Constrúyelo

```python
import numpy as np

def resolver(A, b):
    return np.linalg.solve(A, b)

def lstsq(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]
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

1. Si singular, usar lstsq.
2. Si solucion es grande, sospechar mal condicionamiento.
3. Para sistemas sparse, scipy.sparse.linalg.
4. Para no cuadrados, lstsq.
```

## Ejercicios

1. **Regresion lineal multiple**: con 5 features, ajusta el modelo
   y verifica residuos.
2. **Ridge regression**: implementa con lstsq + regularizacion.
3. **Desafio**: implementa KKT para SVM lineal.

## Lecturas recomendadas

- "Numerical Linear Algebra" (Trefethen & Bau)
- LAPACK: <https://netlib.org/lapack/>

---

> 📚 **Adaptación al español** de la lección "[Linear Systems]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).