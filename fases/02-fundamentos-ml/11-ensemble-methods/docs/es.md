# Metodos de ensemble

> Bagging, boosting, stacking: las tecnicas que mas han ganado competencias de ML en datos tabulares.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-arboles-de-decision-random-forest
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar voto mayoritario.
- Implementar bagging con bootstrap.
- Implementar boosting con pesos.
- Diagnosticar cuando aplicar cada tecnica.

## Constrúyelo

```python
import numpy as np


def voting_clasificacion(predicciones_list):
    predicciones = np.array(predicciones_list)
    valores, counts = np.unique(predicciones, return_counts=True)
    return valores[np.argmax(counts)]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ensemble-elegir
fase: 02
leccion: 11
---

1. Varianza: Bagging.
2. Sesgo: Boosting.
3. Modelos diversos: Stacking.
4. Desbalanceado: EasyEnsemble.
```

## Ejercicios

1. **Stacking**: implementa un meta-modelo que combina 3 modelos base.
2. **Gradient boosting**: implementa el ajuste de pesos via gradiente.
3. **Desafio**: implementa AdaBoost completo con reweighting.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 8
- XGBoost: <https://xgboost.readthedocs.io/>
- LightGBM: <https://lightgbm.readthedocs.io/>

---

> 📚 **Adaptación al español** de la lección "[Ensemble Methods: Boosting, Bagging, Stacking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).