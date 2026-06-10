# Regularización

> Sin regularizacion, las redes modernas sobreajustan en minutos. Dropout, weight decay, batch norm y data augmentation son los cinturones de seguridad.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar dropout forward y backward.
- Implementar L1 y L2 regularization.
- Implementar batch normalization.
- Implementar early stopping.

## Constrúyelo

```python
def dropout_forward(x, p=0.5, entrenamiento=True, semilla=0):
    if not entrenamiento:
        return x, None
    rng = np.random.default_rng(semilla)
    mask = rng.binomial(1, 1 - p, size=x.shape)
    return x * mask / (1 - p), mask  # inverted dropout
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-regularizar
fase: 03
leccion: 07
---

1. Vision: BN + dropout 0.1-0.3 + weight decay 5e-4.
2. NLP: LayerNorm + dropout 0.1 + weight decay 0.01.
3. Tabular MLP: dropout 0.1-0.5 + early stopping.
4. Monitor gap train vs val.
```

## Ejercicios

1. **Dropout en CNN**: implementa SpatialDropout (apaga
   canales enteros, no pixeles).
2. **Label smoothing**: suaviza one-hot para regularizar.
3. **Desafio**: implementa un trainer completo con early
   stopping + weight decay + dropout en un MLP.

## Lecturas recomendadas

- "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting" (Srivastava et al., 2014)
- "Batch Normalization" (Ioffe & Szegedy, 2015)
- "Decoupled Weight Decay" (Loshchilov & Hutter, 2019)

---

> 📚 **Adaptación al español** de la lección "[Regularization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).