# Optimizadores

> Del gradiente puro (SGD) hasta Adam y AdamW, el optimizador es lo que convierte los gradientes en aprendizaje. La eleccion cambia velocidad de convergencia y calidad del minimo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-backpropagation
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar SGD, SGD+Momentum, AdaGrad, RMSProp, Adam.
- Comparar convergencia en una funcion simple.
- Diagnosticar learning rate y momentum.

## Constrúyelo

```python
class Adam:
    def actualizar(self, params, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-optim-elegir
fase: 03
leccion: 06
---

1. Vision: SGD+Momentum lr=0.1.
2. NLP: AdamW lr=1e-4, warmup.
3. Tabular MLP: Adam lr=1e-3.
4. RL: Adam lr=3e-4.
5. LR range test para calibrar.
```

## Ejercicios

1. **AdamW**: implementa weight decay desacoplado.
2. **Lookahead**: k pasos con optimizador interno, luego promedia.
3. **Desafio**: implementa Lion (Chen et al. 2023), sign-based.

## Lecturas recomendadas

- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- "An overview of gradient descent optimization algorithms"
  (Ruder, 2017)

---

> 📚 **Adaptación al español** de la lección "[Optimizers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).