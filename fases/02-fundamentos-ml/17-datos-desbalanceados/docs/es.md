# Datos desbalanceados

> En fraude, churn, enfermedades raras, la clase que importa es 0.1% del dataset. Accuracy es inutil; todo el pipeline debe re-pensarse.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-metricas-y-validacion-cruzada
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar oversampling aleatorio.
- Implementar undersampling aleatorio.
- Calcular class weights.
- Calcular F1 desde matriz de confusion.

## Constrúyelo

```python
def oversampling_simple(X, y, ratio=1.0, semilla=0):
    rng = np.random.default_rng(semilla)
    clases, counts = np.unique(y, return_counts=True)
    mayor = clases[counts.argmax()]
    menor = clases[counts.argmin()]
    n_mayor = counts.max()
    n_menor = counts.min()
    n_objetivo = int(n_mayor * ratio)
    n_replicas = max(0, n_objetivo - n_menor)
    idx_menor = np.where(y == menor)[0]
    if n_replicas > 0:
        idx_nuevos = rng.choice(idx_menor, size=n_replicas, replace=True)
        X = np.vstack([X, X[idx_nuevos]])
        y = np.concatenate([y, y[idx_nuevos]])
    return X, y
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-imbalanced
fase: 02
leccion: 17
---

1. 1:10 a 1:1000: SMOTE + class_weight.
2. Metricas: F1, AUC-PR, recall, MCC.
3. Threshold segun costo.
4. SMOTE SOLO en train.
```

## Ejercicios

1. **SMOTE**: implementa la interpolacion con k-NN.
2. **ADASYN**: SMOTE adaptativo (mas replicas en zonas
   minoritarias).
3. **Desafio**: implementa EasyEnsemble (bagging con N
   undersamplings) y compara con class_weight solo.

## Lecturas recomendadas

- "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
- imbalanced-learn: <https://imbalanced-learn.org/>
- "Learning from Imbalanced Data" (He & Garcia, 2009)

---

> 📚 **Adaptación al español** de la lección "[Imbalanced Data]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).