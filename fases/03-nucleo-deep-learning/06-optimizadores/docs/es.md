# 06 — Optimizadores: SGD, Momentum, Adam, AdamW

> El optimizador decide cómo actualizar los pesos. Adam es la default moderna; entender sus piezas es entender por qué funciona.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-backpropagation
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar SGD con momentum y Nesterov desde cero.
- Implementar Adam y AdamW y entender cada componente.
- Diagnosticar cuándo cada optimizador es mejor.
- Conocer Lion, Sophia y otros optimizadores modernos.

## El problema

Tienes un modelo con millones de parámetros. La pérdida es
una superficie enorme y no convexa. ¿Cómo actualizas los
pesos para minimizar la pérdida eficientemente? SGD puro
es lento y zigzaguea; Adam adapta el learning rate por
parámetro y converge más rápido. La lección cubre los
optimizadores canónicos y cuándo cada uno gana.

## El concepto

**SGD (Stochastic Gradient Descent).** Actualiza los pesos
con el gradiente estimado sobre un mini-batch:

```text
θ ← θ - lr · ∇L_B(θ)
```

Simple, teóricamente bien entendido, pero lento en
superficies curvadas (zigzag) y sensible al learning rate.

**SGD con Momentum.** Agrega una fracción del update
anterior como "inercia":

```text
v ← β · v + ∇L_B(θ)
θ ← θ - lr · v
```

`β ≈ 0.9` es típico. La inercia suaviza el zigzag y acelera
el movimiento en direcciones consistentes.

**Nesterov Accelerated Gradient (NAG).** Variante de
momentum: evalúa el gradiente en el punto "anticipado"
`θ - β · v` en vez del punto actual. Convergen ligeramente
más rápido que momentum clásico.

**AdaGrad.** Adapta el learning rate por parámetro: los
parámetros con gradientes grandes (frecuentes) reciben
learning rate pequeño, los raros reciben grande. Acumula
cuadrados de gradientes en `s`:

```text
s ← s + g ⊙ g
θ ← θ - lr · g / (sqrt(s) + ε)
```

Problema: `s` solo crece, el learning rate efectivo solo
decrece, eventualmente la red deja de aprender. RMSProp
soluciona esto con un decay exponencial.

**RMSProp.** Mantiene una media móvil exponencial de los
cuadrados de gradientes en vez de acumular:

```text
s ← β · s + (1 - β) · g ⊙ g
θ ← θ - lr · g / (sqrt(s) + ε)
```

`β ≈ 0.999` es típico. Resuelve el problema de AdaGrad de
que `s` solo crece.

**Adam (Adaptive Moment Estimation).** Combina momentum y
RMSProp. Mantiene dos medias móviles: la del gradiente
(`m`, primer momento) y la de sus cuadrados (`v`, segundo
momento). Corrige el sesgo inicial con `m_hat, v_hat`.

```text
m ← β_1 · m + (1 - β_1) · g
v ← β_2 · v + (1 - β_2) · g ⊙ g
m_hat = m / (1 - β_1^t)
v_hat = v / (1 - β_2^t)
θ ← θ - lr · m_hat / (sqrt(v_hat) + ε)
```

`β_1 = 0.9`, `β_2 = 0.999`, `lr = 1e-3` son los defaults. Es
el optimizador estándar en deep learning moderno.

**AdamW (Adam with decoupled Weight decay).** Separa el
weight decay del gradiente. En Adam, el weight decay
interactúa con la adaptación del learning rate, lo que
empeora la regularización. AdamW aplica `θ ← θ - lr · λ · θ`
explícitamente, fuera de la adaptación:

```text
θ ← θ - lr · (m_hat / (sqrt(v_hat) + ε) + λ · θ)
```

Es el default en transformers modernos. Loafin fine-tuning
de LLMs es prácticamente siempre con AdamW.

**Lion (EvoLved Sign Momentum).** Sign-based optimizer
descubierto por symbolic search. Usa solo el signo del
gradiente (no la magnitud):

```text
update = sign(β_1 · m + (1 - β_1) · g)
m ← β_2 · m + (1 - β_2) · g
θ ← θ - lr · (update + λ · θ)
```

Más eficiente en memoria que Adam (solo un momento).
Usado en algunos LLMs recientes.

**Sophia.** Second-order optimizer que aproxima la
Hessiana con un estimador de diagonal Hutchinson. Más
rápido que Adam en LLMs grandes.

**Cuándo usar cada optimizador.**

| Optimizer | Cuándo |
|---|---|
| SGD | Baseline, control fino, computer vision |
| SGD + momentum | Computer vision, datasets grandes |
| Adam | Default para transformers, RL, GANs |
| AdamW | Fine-tuning de LLMs (siempre) |
| Lion | Alternativa a AdamW, menos memoria |
| Sophia | LLMs grandes con presupuesto ajustado |

**Trampas.**

- **SGD con learning rate fijo:** no converge bien. Usa
  schedule o SGD con warmup + cosine decay.
- **Adam con weight decay alto:** la regularización
  interactúa mal con la adaptación. Usa AdamW.
- **Sin bias correction en Adam al inicio:** el
  `m, v` están sub-estimados, los updates son
  artificialmente pequeños. Siempre incluye `m_hat, v_hat`.
- **Cambiar optimizer a mitad de entrenamiento:** puede
  romper la convergencia. Comprométete a uno.

## Constrúyelo

```python
import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] + g
            update = (self.momentum * self.v[i] + g
                      if self.nesterov else self.v[i])
            p -= self.lr * update


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            # Decoupled weight decay
            p -= self.lr * self.weight_decay * p
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
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
name: prompt-optimizer
fase: 03
leccion: 06
---

Eres un asistente que ayuda a elegir y configurar el
optimizador. Recibirás la arquitectura del modelo, la tarea,
y los síntomas del entrenamiento. Tu trabajo:

1. Transformer, GAN, RL: Adam (lr=1e-3).
2. Fine-tuning de LLM: AdamW (lr=2e-5 a 1e-4, wd=0.01).
3. Computer vision: SGD + momentum (lr=0.1, momentum=0.9)
   o AdamW.
4. Si los gradientes explotan: gradient clipping a 1.0.
5. Si los gradientes son 0: aumentar lr o cambiar a ReLU.
6. Si la loss no baja: scheduler con warmup + cosine.
7. Si quieres máximo control: SGD con momentum y schedule
   manual.
8. Recomienda monitorear el gradiente norm total y
   individual por capa.
```

## Ejercicios

1. **SGD vs Adam**: entrena un MLP pequeño con ambos y
   compara la curva de loss en un dataset de juguete.
2. **AdamW**: implementa la versión con decoupled weight
   decay y compara con Adam puro.
3. **Desafío**: implementa Lion y compara memoria y
   convergencia con AdamW en un mini-transformer.

## Lecturas recomendadas

- *Adam: A Method for Stochastic Optimization* — Kingma &
  Ba, 2014.
- *Decoupled Weight Decay Regularization* — Loshchilov &
  Hutter, 2019.
- *Symbolic Discovery of Optimization Algorithms* — Chen
  et al., 2023 (Lion).
- PyTorch optim: <https://pytorch.org/docs/stable/optim.html>.

---

> 📚 **Adaptación al español** de la lección "[Optimizers]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
