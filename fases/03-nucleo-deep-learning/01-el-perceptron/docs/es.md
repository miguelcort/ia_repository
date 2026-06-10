# El perceptron

> El primer algoritmo que aprendio de datos (Rosenblatt, 1958). Limitado a separables lineales, pero el bloque constructor de todo lo que vino despues.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-fundamentos-ml/02-modelos-lineales-y-regresion-logistica
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar el perceptron con regla de aprendizaje.
- Verificar que aprende AND y OR pero no XOR.
- Diagnosticar datasets linealmente separables.

## Constrúyelo

```python
class Perceptron:
    def fit(self, X, y):
        for epoca in range(self.n_epocas):
            for xi, yi in zip(X, y):
                y_pred = self.predecir(xi)
                if y_pred != yi:
                    self.pesos += self.lr * (yi - y_pred) * xi
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-perceptron
fase: 03
leccion: 01
---

1. ¿Separable? Si no, usar MLP o SVM.
2. lr entre 0.01-1.0, monitorear errores.
3. Practica: regresion logistica > perceptron.
```

## Ejercicios

1. **Pocket algorithm**: guardar los mejores pesos vistos
   durante el entrenamiento.
2. **Voted perceptron**: promediar todos los pesos vistos.
3. **Desafio**: visualizar la frontera de decision en 2D para
   AND, OR, NAND, XOR.

## Lecturas recomendadas

- "Perceptron" (Rosenblatt, 1958)
- "Perceptrons" (Minsky & Papert, 1969) — limitaciones
- "Learning representations by back-propagating errors"
  (Rumelhart, Hinton, Williams, 1986)

---

> 📚 **Adaptación al español** de la lección "[The Perceptron]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).