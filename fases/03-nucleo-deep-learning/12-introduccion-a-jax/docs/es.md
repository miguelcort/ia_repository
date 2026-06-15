# 12 — Introducción a JAX

> JAX es NumPy + grad + jit + vmap + pmap. Es la base de muchos frameworks modernos (Flax, Haiku, Equinox) y de los papers de DeepMind.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-introduccion-a-pytorch
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Escribir funciones NumPy-like en JAX y entender
  `jax.grad`, `jax.jit`, `jax.vmap`, `jax.pmap`.
- Diagnosticar cuándo JAX es preferible a PyTorch.
- Implementar un MLP mínimo con JAX.
- Conocer los frameworks derivados: Flax, Haiku, Equinox.

## El problema

PyTorch es dominante pero JAX ofrece una combinación
única: API NumPy-like (familiar), diferenciación
automática funcional, compilación JIT con XLA para
rendimiento cercano a C++, y paralelización trivial sobre
TPUs. La lección cubre la API esencial de JAX para que
puedas leer código de DeepMind y de los LLMs de Google.

## El concepto

**¿Qué es JAX?** Una biblioteca de Google que extiende
NumPy con:

- `grad`: diferenciación automática de funciones.
- `jit`: compilación Just-In-Time con XLA (acelerador de
  Google).
- `vmap`: vectorización automática sobre un eje.
- `pmap`: paralelización sobre múltiples dispositivos
  (TPUs/GPUs).

API `jax.numpy` es compatible con NumPy al 90%. Funciones
escritas en `jnp` se pueden compilar y ejecutar en CPU,
GPU o TPU sin cambios.

**Pureza funcional.** JAX requiere **funciones puras**:
sin efectos secundarios, sin estado mutable global. Esto
permite a `jit` razonar sobre el código y compilarlo
eficientemente. Es una restricción que disciplina el
código.

**`grad` — diferenciación automática.** Dada una función
`f(x)`, `jax.grad(f)(x)` devuelve `df/dx`. Funciona por
reverse-mode AD (backprop):

```python
import jax.numpy as jnp
from jax import grad

def loss(w, x, y):
    return jnp.mean((x @ w - y) ** 2)

grad_loss = grad(loss)
```

**`jit` — compilación JIT.** Compila una función con XLA
para que se ejecute eficientemente. La primera llamada
es lenta (compilación), las siguientes son rápidas:

```python
from jax import jit

fast_loss = jit(loss)
# Primera llamada: compilación, lenta
# Segunda llamada: ejecutada, rápida
```

**`vmap` — vectorización automática.** Transforma una
función que opera sobre un solo ejemplo en una función
que opera sobre un batch, sin escribir loops:

```python
from jax import vmap

def predict(w, x):  # x: (d,)
    return x @ w  # escalar

predict_batch = vmap(predict, in_axes=(None, 0))
# predict_batch(w, X)  # X: (batch, d), resultado: (batch,)
```

**`pmap` — paralelización multi-dispositivo.** Replica la
función sobre N dispositivos y procesa batches en paralelo.
Para TPUs es la opción natural. En GPUs con peer access
también funciona.

**`random` — PRNG explícito.** A diferencia de NumPy,
JAX requiere pasar una `key` explícita a cada función
aleatoria. Esto hace los experimentos reproducibles:

```python
from jax import random
key = random.PRNGKey(42)
key, subkey = random.split(key)
weights = random.normal(subkey, (d_in, d_out))
```

**Frameworks sobre JAX.**

- **Flax:** estilo PyTorch, define `nn.Module` con
  parámetros. Es el más usado en DeepMind.
- **Haiku:** estilo Sonnet/functional de DeepMind. Las
  funciones se decoran con `@hk.transform`.
- **Equinox:** minimalista, ~1k líneas. Transparente,
  fácil de debuggear.
- **Optax:** optimizadores y schedules listos para usar.

**Cuándo usar JAX vs PyTorch.**

| Aspecto | PyTorch | JAX |
|---|---|---|
| Curva de aprendizaje | Baja (más Python) | Media (funcional) |
| TPU support | Limitado | Excelente (nativo) |
| Compilación JIT | Sí (torch.compile) | Sí (XLA) |
| Debugging | Fácil (Python) | Más difícil (asíncrono) |
| Comunidad | Masiva | Grande, creciendo |
| Reproducibilidad | Manual | Explícita por design |
| Frameworks | Lightning, etc. | Flax, Haiku, Equinox |

**Trampas.**

- **Funciones con efectos secundarios:** `jit` no las
  soporta. Usa `pure_callback` si necesitas I/O.
- **Shapes estáticas:** `jit` requiere shapes conocidas
  en tiempo de compilación. Usa `vmap` para batches
  variables en una dimensión.
- **PRNG key reutilizada:** no se actualiza al usarla.
  Siempre haz `key, subkey = random.split(key)` antes de
  usar.

## Constrúyelo

```python
# Demo conceptual de JAX (no se ejecuta sin instalar jax).
# Para correr: pip install jax jaxlib
import jax.numpy as jnp
from jax import grad, jit, vmap, random


def init_params(key, d_in, d_hidden, d_out):
    keys = random.split(key, 3)
    W1 = random.normal(keys[0], (d_in, d_hidden)) * 0.1
    b1 = jnp.zeros(d_hidden)
    W2 = random.normal(keys[1], (d_hidden, d_out)) * 0.1
    b2 = jnp.zeros(d_out)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def loss_fn(params, x, y):
    logits = forward(params, x)
    return jnp.mean((logits - y) ** 2)


def main():
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    params = init_params(init_key, 2, 16, 1)

    # Datos de juguete
    X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = jnp.array([0.0, 1.0, 1.0, 0.0])

    # Gradiente via grad
    grads = grad(loss_fn)(params, X, y)

    # Versión compilada
    fast_update = jit(
        lambda p, x, y: {k: v - 0.01 * grads[k] for k, v in p.items()}
    )
    new_params = fast_update(params, X, y)

    # Batch via vmap
    predict_batch = vmap(forward, in_axes=(None, 0))
    predictions = predict_batch(new_params, X)

    print(f"Pérdida: {loss_fn(params, X, y):.3f}")
    print(f"Predicciones: {predictions}")


if __name__ == "__main__":
    main()
```

## Úsalo

```bash
pip install jax jaxlib
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-jax-elegir
fase: 03
leccion: 12
---

Eres un asistente que ayuda a elegir entre JAX y PyTorch.
Recibirás la arquitectura, el hardware disponible, y el
contexto del equipo. Tu trabajo:

1. Si vas a entrenar en TPU: JAX (Flax o Haiku).
2. Si necesitas speedup sin reescribir: jax.jit sobre
   código existente.
3. Si el equipo viene de TensorFlow/Sonnet: Haiku.
4. Si vienes de PyTorch y no necesitas TPU: PyTorch es más
   fácil.
5. Si haces research en optimización numérica o
   diferenciación: JAX.
6. Recomendar Flax para producción sobre JAX puro.
7. Advertir: JAX requiere funciones puras; migrar código
   con estado requiere refactor.
8. Para debugging: Equinox es más transparente.
```

## Ejercicios

1. **MLP en JAX**: implementa un MLP de 2 capas para
   clasificar XOR usando `grad` y `jit`.
2. **vmap vs loop**: compara el speedup de `vmap` sobre un
   MLP batch vs un loop explícito.
3. **Desafío**: implementa un optimizador AdamW en JAX puro
   (sin Optax).

## Lecturas recomendadas

- *JAX: Accelerated Machine Learning Research* — Bradbury
  et al., 2018 (whitepaper).
- JAX docs: <https://jax.readthedocs.io>.
- Flax: <https://flax.readthedocs.io>.
- Equinox: <https://docs.equinox.uk>.
- Optax: <https://optax.readthedocs.io>.

---

> 📚 **Adaptación al español** de la lección "[Introduction to JAX]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
