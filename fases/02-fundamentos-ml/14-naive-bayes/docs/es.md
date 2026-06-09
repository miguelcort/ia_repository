# Naive Bayes

> El clasificador mas viejo de ML sigue siendo relevante: rapido, robusto, base para spam, sentimiento y categorizacion de texto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar GaussianNB (continuas, asumiendo normalidad).
- Implementar MultinomialNB (conteos, con Laplace smoothing).
- Diagnosticar cuando NB es la opcion correcta.

## Constrúyelo

```python
import numpy as np


class GaussianNB:
    def fit(self, X, y):
        self.clases = np.unique(y)
        self.mu = np.array([X[y == c].mean(axis=0) for c in self.clases])
        self.sigma = np.array([X[y == c].std(axis=0) + 1e-9 for c in self.clases])
        return self

    def predict(self, X):
        # ... log P(x|y) + log P(y)
        return ...
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nb-elegir
fase: 02
leccion: 14
---

1. Continuas: GaussianNB.
2. Conteos: MultinomialNB.
3. Binarias: BernoulliNB.
4. Desbalanceado: ComplementNB.
```

## Ejercicios

1. **BernoulliNB**: implementa la version para features binarias.
2. **TF-IDF + MultinomialNB**: clasificador de spam completo.
3. **Desafio**: compara NB vs LogReg en el dataset 20newsgroups.

## Lecturas recomendadas

- "Speech and Language Processing" (Jurafsky & Martin), cap. 4
- scikit-learn naive_bayes: <https://scikit-learn.org/stable/modules/naive_bayes.html>

---

> 📚 **Adaptación al español** de la lección "[Naive Bayes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).