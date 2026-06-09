# Deteccion de anomalias

> Fraude, fallas, intrusiones: anomalias son raras por definicion. Las tecnicas clasicas (Z-score, IQR) sorprenden en muchos casos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Detectar outliers univariados con Z-score e IQR.
- Detectar outliers multivariados con Mahalanobis.
- Implementar un Isolation Forest simplificado.

## Constrúyelo

```python
def zscore_anomalias(x, umbral=3.0):
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return np.abs((x - mu) / sigma) > umbral
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-anomalia-elegir
fase: 02
leccion: 16
---

1. 1D sin correlacion: Z/IQR.
2. Multivariado Gaussiano: Mahalanobis.
3. Multivariado no parametrico: IForest/LOF.
4. Series: STL residual.
5. Alta dim: Autoencoder.
```

## Ejercicios

1. **LOF**: implementa Local Outlier Factor (densidad local).
2. **One-Class SVM**: implementa la frontera con kernel RBF.
3. **Desafio**: Autoencoder con error de reconstruccion como
   score; visualiza en MNIST las reconstrucciones.

## Lecturas recomendadas

- "Isolation Forest" (Liu, Ting, Zhou — ICDM 2008)
- "LOF: Identifying Density-Based Local Outliers" (Breunig et al.)
- PyOD: <https://pyod.readthedocs.io/>

---

> 📚 **Adaptación al español** de la lección "[Anomaly Detection]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).