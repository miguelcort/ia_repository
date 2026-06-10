# Funciones de pérdida

> La loss es la funcion objetivo: define que significa "aprender bien". La eleccion correcta de loss + output activation + problema = convergencia limpia.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar MSE, MAE, BCE, CCE, Huber, contrastive.
- Conocer que loss usar para cada problema.
- Diagnosticar NaN/Inf y estabilizar numéricamente.

## Constrúyelo

```python
def bce(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-loss-elegir
fase: 03
leccion: 05
---

1. Regresion: MSE/MAE/Huber.
2. Binaria: BCE + sigmoid.
3. Multiclase: CCE + softmax.
4. Metrica: contrastive / triplet.
5. BCE/CCE: clip a [eps, 1-eps].
```

## Ejercicios

1. **Focal loss**: extension de CCE para desbalance.
2. **Triplet loss**: para embeddings (anchor, positive, negative).
3. **Desafio**: implementa IoU/Dice loss para segmentacion.

## Lecturas recomendadas

- "Cross-Entropy vs. MSE" — discussion historica
- "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- "Dimensionality Reduction by Learning an Invariant Mapping"
  (Hadsell et al., 2006) — contrastive

---

> 📚 **Adaptación al español** de la lección "[Loss Functions]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).