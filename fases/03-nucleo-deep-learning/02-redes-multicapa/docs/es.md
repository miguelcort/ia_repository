# Redes multi-capa y forward pass

> Componer perceptrones con no-linealidades es la base de deep learning. Universal approximation theorem: con 1 capa oculta + no-linealidad, puedes aproximar cualquier funcion continua.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-el-perceptron
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar una capa densa (fully-connected).
- Implementar activaciones (sigmoid, tanh, ReLU).
- Encadenar capas en una red feedforward.
- Verificar formas de salida y gradiente.

## Constrúyelo

```python
class CapaDensa:
    def forward(self, x):
        z = x @ self.W + self.b
        return sigmoid(z)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mlp-arquitectura
fase: 03
leccion: 02
---

1. Tabular: 2-3 capas, 64-256, ReLU, dropout, BN.
2. Output: sigmoid/softmax/lineal.
3. Empezar pequeno, crecer si underfit.
```

## Ejercicios

1. **Softmax**: implementa softmax estable (numericamente
   estable restando el max).
2. **MLP funcional**: entrena la red con backprop manual
   (sin frameworks) sobre MNIST.
3. **Desafio**: implementa learning rate decay y early
   stopping.

## Lecturas recomendadas

- "Deep Learning" (Goodfellow, Bengio, Courville) cap. 6
- "Neural Networks and Deep Learning" (Michael Nielsen) cap. 1-2

---

> 📚 **Adaptación al español** de la lección "[Multi-Layer Networks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).