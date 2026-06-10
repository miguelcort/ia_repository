# Depuración de redes neuronales

> El 80% del tiempo entrenando redes se va en debug. Saber usar gradient check, detectar NaN, monitorear pesos y diagnosticar over/underfitting es la diferencia entre terminar y abandonar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-programacion-de-learning-rate
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar gradient check numerico.
- Detectar NaN/Inf en tensores.
- Monitorear estadisticas de pesos por capa.

## Constrúyelo

```python
def check_gradiente_numerico(f, grad_analitico, x, h=1e-5):
    grad_num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad_num[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return float(np.abs(grad_analitico - grad_num).max())
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-debug-red
fase: 03
leccion: 13
---

1. Loss no baja: overfit mini-batch.
2. NaN: gradient clipping, clip BCE, normalizar.
3. Overfit: dropout, weight decay, early stopping.
4. Underfit: mas capacidad, menos regularizacion.
5. TensorBoard siempre.
```

## Ejercicios

1. **Visualizar gradientes**: plot ||g|| por capa vs
   epoca. Detectar vanishing/exploding.
2. **Learning rate finder**: lr de 1e-7 a 10, plotear
   loss, encontrar max lr estable.
3. **Desafio**: implementar un trainer con TensorBoard
   logging (SummaryWriter).

## Lecturas recomendadas

- "A Recipe for Training Neural Networks" (Karpathy, 2019)
- "Why gradient clipping accelerates training" (Zhang et al.)
- TensorBoard: <https://www.tensorflow.org/tensorboard>

---

> 📚 **Adaptación al español** de la lección "[Debugging Neural Networks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).