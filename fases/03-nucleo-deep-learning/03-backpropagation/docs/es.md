# Backpropagation desde cero

> La invencion que permitio deep learning: calcular el gradiente de toda la red en una sola pasada, propagando errores hacia atras. Sin esto, entrenar redes profundas era inviable.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar backprop completo en una red feedforward.
- Verificar que aprende XOR (imposible para perceptrón).
- Diagnosticar gradientes desvanecientes/explosivos.

## Constrúyelo

```python
def backward(self, y_true, y_pred):
    n = len(y_true)
    grad = 2 * (y_pred - y_true) / n * sigmoid_derivada(self.capas[-1]["z"])
    for i in reversed(range(len(self.capas))):
        capa = self.capas[i]
        capa["grad_W"] = (capa["a" if i > 0 else "x"]).T @ grad
        capa["grad_b"] = grad.sum(axis=0)
        if i > 0:
            grad = grad @ capa["W"].T * sigmoid_derivada(self.capas[i - 1]["z"])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-backprop-debug
fase: 03
leccion: 03
---

1. Loss no baja: lr muy bajo, arquitectura debil.
2. NaN: exploding grad. Gradient clipping.
3. Solo capa final aprende: vanishing. ReLU+He, ResNet.
4. Gradient check: numerico vs analitico.
```

## Ejercicios

1. **Gradient check**: implementa verificacion numerica.
2. **Backprop para softmax + cross-entropy**: simplifica la
   ultima capa derivando a mano.
3. **Desafio**: entrena una red de 5 capas en MNIST y observa
   gradientes capa por capa.

## Lecturas recomendadas

- "Learning representations by back-propagating errors"
  (Rumelhart, Hinton, Williams, 1986)
- "Yes you should understand backprop" (Karpathy, 2016)
- "Calculus on Computational Graphs" (Christopher Olah)

---

> 📚 **Adaptación al español** de la lección "[Backpropagation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).