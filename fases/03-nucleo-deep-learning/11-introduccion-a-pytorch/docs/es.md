# Introducción a PyTorch

> El framework default en deep learning moderno. Tensores con autograd, modulos componibles, optimizadores listos, GPU transparente. Una vez que lo conoces, el resto es composicion.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-mini-framework
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Conocer la API basica: Tensor, nn.Module, optim, DataLoader.
- Implementar un loop de entrenamiento completo.
- Diagnosticar cuando usar GPU y mixed precision.

## Constrúyelo

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(modelo.parameters(), lr=0.05)

for epoca in range(500):
    y_pred = modelo(X)
    loss = loss_fn(y_pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pytorch-pattern
fase: 03
leccion: 11
---

1. Tabular: nn.Sequential Linear+ReLU+BN+Dropout.
2. Vision: Conv+BN+ReLU+MaxPool.
3. NLP: Embedding+LSTM/Transformer.
4. Loop: zero_grad, forward, loss, backward, step.
5. Inference: .eval() + torch.no_grad().
```

## Ejercicios

1. **Dataset propio**: implementa torch.utils.data.Dataset
   para imagenes.
2. **Training loop completo**: anade validation, scheduler,
   early stopping.
3. **Desafio**: entrena un MLP en MNIST con GPU y mixed
   precision (torch.cuda.amp).

## Lecturas recomendadas

- PyTorch tutorials: <https://pytorch.org/tutorials/>
- "Deep Learning with PyTorch" (Stevens, Antiga, Viehmann)
- PyTorch Lightning: <https://lightning.ai/>

---

> 📚 **Adaptación al español** de la lección "[Introduction to PyTorch]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).