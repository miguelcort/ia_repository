# Seleccion de features

> Mas features no siempre es mejor: malgasta capacidad del modelo, introduce ruido y rompe la interpretabilidad. Saber reducir dimensionalidad es saber.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-preparacion-de-datos-y-feature-engineering
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar varianza threshold.
- Implementar eliminacion por correlacion.
- Calcular mutual information discreta.
- Implementar forward selection.

## Constrúyelo

```python
def varianza_threshold(X, umbral=0.0):
    var = X.var(axis=0)
    mascara = var > umbral
    return mascara, var
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-feature-selection
fase: 02
leccion: 18
---

1. Varianza threshold.
2. Correlacion (>0.95).
3. Filter (chi2, ANOVA, MI).
4. Embedded (Lasso).
5. Wrapper (RFE).
6. Validar con CV.
```

## Ejercicios

1. **Lasso path**: implementa la regularizacion L1 y observa
   como coeficientes caen a cero.
2. **Boruta**: implementa la seleccion robusta basada en RF
   shadow features.
3. **Desafio**: RFE con cross-validation sobre un dataset
   realista (e.g. MNIST reducido).

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 6
- "Feature Selection for Data Mining" (Liu & Motoda)
- scikit-learn feature_selection: <https://scikit-learn.org/stable/modules/feature_selection.html>

---

> 📚 **Adaptación al español** de la lección "[Feature Selection]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).