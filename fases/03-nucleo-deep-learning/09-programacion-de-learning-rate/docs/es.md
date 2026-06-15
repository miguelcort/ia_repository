# 09 — Schedules de learning rate y warmup

> El learning rate es el hiperparámetro más importante. Un buen schedule es la diferencia entre converger y divergir.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-optimizadores
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar step, exponential y cosine decay schedules.
- Implementar warmup para estabilizar el inicio del
  entrenamiento.
- Diagnosticar cuándo usar cada schedule.
- Conocer schedules modernos: cosine with restarts, OneCycle.

## El problema

Entrenas con `lr = 0.1` y la loss diverge al primer paso.
Bajas a `lr = 0.001` y la red aprende pero lentamente.
El learning rate perfecto varía a lo largo del
entrenamiento: alto al principio para explorar, bajo al
final para converger. La lección cubre los schedules
canónicos (step, exponential, cosine) y la práctica moderna
de warmup + cosine decay.

## El concepto

**Step decay.** Reduce el learning rate por un factor (típico
0.1) cada N epochs. Simple pero poco elegante. Ejemplo:
`lr = lr_0 * 0.1 ** (epoch // 30)`.

**Exponential decay.** `lr = lr_0 * γ^epoch` con `γ ≈ 0.95`.
Decae continuamente, no por escalones. Más suave que step
decay.

**Cosine decay.** `lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + cos(π t / T))`.
Decae siguiendo una coseno de `lr_0` a `lr_min` en `T`
pasos. Es la default moderna: suave al principio y al final,
agresivo en el medio.

**Warmup.** Las primeras N iteraciones incrementan el
learning rate linealmente de 0 (o un valor pequeño) a
`lr_0`. Previene divergencia al inicio cuando los pesos
aleatorios producen gradientes grandes. Crítico en
transformers (BERT, GPT) y en fine-tuning de LLMs.

**Linear warmup + cosine decay.** La combinación estándar
en transformers:

```text
if t < warmup_steps:
    lr = lr_0 * t / warmup_steps
else:
    t' = (t - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + cos(π t'))
```

Típico: `warmup_steps = 0.01 * total_steps` (1% del
entrenamiento). Para LLMs, hasta 5%.

**OneCycle policy (Smith, 2018).** Un ciclo completo: lr
sube de `lr_0/div_factor` a `lr_max` en la primera mitad
del entrenamiento, luego baja a `lr_0/div_factor` en la
segunda mitad. El momentum sigue el patrón inverso. Da
convergencia más rápida que cosine decay fijo.

**Cosine with restarts (SGDR).** Cosine decay pero
reiniciando el learning rate al máximo periódicamente.
Permite al modelo escapar mínimos locales. Cada restart
"sacrifica" convergencia por exploración.

**Linear decay (simple).** `lr = lr_0 * (1 - t / T)`. Decae
linealmente a 0. Simple, predecible, suficiente en muchos
casos.

**Cuándo usar cada schedule.**

| Schedule | Cuándo |
|---|---|
| Step | Baseline simple, datasets pequeños |
| Exponential | Datasets tabulares clásicos |
| Cosine | Default para transformers, LLMs |
| Linear | Alternativa simple a cosine |
| OneCycle | Cuando quieres entrenar rápido y probar |
| Cosine with restarts | RNNs, GANs (escapar mínimos) |
| Warmup + cosine | Transformers, fine-tuning |

**Hiperparámetros clave.**

- `lr_0`: lr máximo. Default: 1e-3 (Adam), 1e-4 a 2e-5
  (AdamW para fine-tuning de LLMs).
- `warmup_steps`: típico 100-1000 o 1-5% del total.
- `lr_min`: 1e-5 a 1e-6. Si es 0, el lr llega a 0 al final.

**Trampas.**

- **Sin warmup con Adam + learning rate alto:** divergencia
  inmediata. Siempre warmup para transformers y LLMs.
- **Warmup demasiado largo:** el modelo no aprende
  rápidamente al principio.
- **Schedule muy agresivo al final:** lr llega a 0 antes
  de converger. Usar `lr_min > 0` o no terminar en 0.
- **Cambiar el schedule a mitad de entrenamiento:** no
  funciona. El schedule define el "presupuesto" total.

## Constrúyelo

```python
import math


def constant_lr(lr):
    return lambda step: lr


def step_decay(lr_init, drop=0.1, step_size=30):
    def schedule(step):
        return lr_init * drop ** (step // step_size)
    return schedule


def exponential_decay(lr_init, gamma=0.95):
    def schedule(step):
        return lr_init * gamma ** step
    return schedule


def cosine_decay(lr_init, lr_min=0, total_steps=1000):
    def schedule(step):
        t = min(step, total_steps) / total_steps
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * t))
    return schedule


def warmup_cosine(lr_init, lr_min=0, warmup_steps=100, total_steps=1000):
    """Linear warmup + cosine decay. Default en transformers."""
    def schedule(step):
        if step < warmup_steps:
            return lr_init * step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * t))
    return schedule


def onecycle(lr_max, total_steps, div_factor=25, pct_start=0.3):
    """OneCycle: lr sube en pct_start, luego baja."""
    def schedule(step):
        if step < pct_start * total_steps:
            t = step / (pct_start * total_steps)
            return lr_max * (1 + t * (div_factor - 1)) / div_factor
        else:
            t = (step - pct_start * total_steps) / (
                (1 - pct_start) * total_steps
            )
            return lr_max * (1 - t) * (div_factor - 1) / div_factor
    return schedule
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-lr-schedule
fase: 03
leccion: 09
---

Eres un asistente que ayuda a elegir y configurar el schedule
de learning rate. Recibirás la arquitectura, el optimizador,
y los síntomas (loss diverge al inicio, no converge, etc.).
Tu trabajo:

1. Si transformer o LLM: warmup (1-5% de steps) + cosine
   decay. lr=1e-4 (AdamW) para fine-tuning, 1e-3 para
   from-scratch.
2. Si CNN con SGD: step decay cada 30 epochs, factor 0.1.
3. Si quieres entrenar rápido: OneCycle.
4. Si la loss diverge al inicio: warmup más largo o lr
   más bajo.
5. Si la loss no converge: el schedule es demasiado
   agresivo, lr_min más alto.
6. Si lr llega a 0 antes de converger: no terminar en 0.
7. Recomienda monitorear lr durante entrenamiento.
```

## Ejercicios

1. **Step vs cosine**: entrena un MLP pequeño con ambos
   y compara la curva de loss.
2. **Warmup**: visualiza el efecto del warmup en la
   estabilidad del entrenamiento.
3. **Desafío**: implementa cosine with warm restarts
   (SGDR) y compara convergencia con cosine simple.

## Lecturas recomendadas

- *SGDR: Stochastic Gradient Descent with Warm Restarts* —
  Loshchilov & Hutter, 2017.
- *A Disciplined Approach to Neural Network Hyper-
  Parameters* — Smith, 2018 (OneCycle).
- PyTorch optim.lr_scheduler: <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>.

---

> 📚 **Adaptación al español** de la lección "[Learning Rate Schedules]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
