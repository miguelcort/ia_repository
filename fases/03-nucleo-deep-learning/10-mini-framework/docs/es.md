# Mini-framework de deep learning

> Construir tu propio framework (Modulo, Sequential, Optim) es el mejor ejercicio para entender que hace PyTorch. Una vez que lo entiendes, los frameworks dejan de ser magia.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-programacion-de-learning-rate
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la clase base `Modulo` con forward/backward.
- Construir Linear, ReLU, Sigmoid como módulos.
- Encadenarlos en `Sequential`.
- Construir un `Entrenador` con loop completo.

## Constrúyelo

```python
class Modulo:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError


class Linear(Modulo):
    def forward(self, x):
        self.cache_x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.grad_W = self.cache_x.T @ grad
        self.grad_b = grad.sum(axis=0)
        return grad @ self.W.T
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ml-internals
fase: 03
leccion: 10
---

1. Default: PyTorch.
2. Investigacion: JAX/TPU.
3. Mobile: TFLite/PyTorch Mobile.
4. Servidor: TorchServe, BentoML, Triton.
5. HF Transformers, PyTorch Lightning, Keras.
```

## Ejercicios

1. **Convolucion 2D**: implementa Conv2d (forward + backward).
2. **Cross-entropy + softmax**: modulo combinado (mas estable).
3. **Desafio**: implementa autograd basico (registrar ops en
   un DAG y llamar backward sobre el grafo).

## Lecturas recomendadas

- PyTorch internals: <https://pytorch.org/docs/stable/notes/extending.html>
- "PyTorch Under the Hood" (Edward Yang, 2020)
- JAX docs: <https://jax.readthedocs.io/>

---

> 📚 **Adaptación al español** de la lección "[Mini Framework]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).