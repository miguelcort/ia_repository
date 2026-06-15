# 13 — Depuración de redes neuronales

> Una red neuronal que no entrena es la pesadilla más común. Saber diagnosticar y arreglar es la habilidad que separa a los engineers de los demás.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-optimizadores,
                  09-programacion-de-learning-rate
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Diagnosticar loss que no baja, loss que explota, y
  gradientes muertos.
- Aplicar gradient clipping, learning rate finder y
  debugging tools.
- Usar TensorBoard o W&B para visualizar entrenamiento.
- Implementar un check pre-flight antes de entrenar.

## El problema

Entrenas tu red. La loss se queda en `2.3` (chance level) o
explota a `NaN`. Los gradientes son `0` o `1e10`. ¿Qué
revisas primero? La lección da un flujo de diagnóstico
ordenado y las soluciones canónicas a cada modo de falla.

## El concepto

**El flujo de diagnóstico.** Cuando algo va mal, sigue
este orden:

1. **Overfit a un mini-batch.** Entrena con un batch de
   32 ejemplos hasta loss ~0. Si no converge, el bug está en
   el modelo, los datos o el loop de entrenamiento.
2. **Verifica shapes y dtypes.** Un broadcasting mal
   aplicado o un dtype mixto es el bug más común.
3. **Visualiza gradientes.** Imprime su norma por capa
   en cada step. ¿Son todos similares? ¿Algunos son 0?
4. **Visualiza activaciones.** Igual con la varianza de
   las activaciones por capa.
5. **Verifica el data pipeline.** ¿Las etiquetas son
   correctas? ¿El augmentación no es demasiado agresivo?
6. **Reduce el learning rate.** El 80% de los problemas
   desaparecen con `lr / 10`.

**Modos de falla canónicos y soluciones.**

| Síntoma | Causa probable | Solución |
|---|---|---|
| Loss no baja | LR muy bajo, bug en loop | Aumentar LR, verificar shapes |
| Loss explota (NaN) | LR muy alto | Reducir LR, gradient clipping |
| Loss NaN desde step 0 | Bug en datos (inf, NaN) | Verificar input pipeline |
| Loss oscila | LR demasiado alto, batch pequeño | Reducir LR, aumentar batch |
| Train loss << val loss | Overfitting | Regularización, augmentation, early stop |
| Train loss = val loss, ambos altos | Underfitting | Modelo más grande, más datos, más epochs |
| Gradientes = 0 | Dying ReLU, LR muy bajo, exploding→clipped | Cambiar activación, LR, no clip tan agresivamente |
| Gradientes explotan | LR alto, no hay clipping | Gradient clipping, normalización |
| Loss plateau temprano | LR decay demasiado agresivo | Ajustar schedule |

**Gradient clipping.** Limita la norma del gradiente:

```python
grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
# O clipping por norma total:
grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(grads)))
scale = jnp.minimum(1.0, max_norm / grad_norm)
grads = jax.tree_map(lambda g: g * scale, grads)
```

Típico: `max_norm = 1.0`. Crítico en entrenamiento de
LLMs.

**Learning rate finder (Smith, 2017).** Empieza con LR muy
bajo (1e-7), incrementándolo exponencialmente por N
steps, y grafica la loss. El LR óptimo está justo antes de
que la loss explote. Implementación: `torch_lr_finder` o
similar.

**Weight initialization check.** Imprime la norma de
cada capa de pesos al inicio. Si las normas son muy
diferentes entre capas, hay desbalance. Si todas son ~0,
la inicialización colapsó.

**Activation statistics.** Imprime la media y varianza
de las activaciones de cada capa en cada step. Si una
capa tiene varianza ~0, sus neuronas están saturadas
(sigmoid) o muertas (ReLU).

**Overfit a un mini-batch.** El primer test de sanidad.
Si tu modelo no puede bajar la loss a ~0 sobre 32
ejemplos memorizados, el problema es fundamental (bug,
incompatibilidad, mala inicialización).

**TensorBoard / W&B.** Visualización de:
- Loss de train y val.
- Norma de gradientes por capa.
- Distribución de pesos y activaciones.
- Learning rate actual.
- Imágenes de predicciones vs realidad.

**Dead neuron diagnosis.** Si una neurona ReLU siempre
emite 0, su gradiente siempre es 0, y nunca se recupera.
Mide la fracción de neuronas "muertas" durante el
entrenamiento. Si supera el 50%, cambia a leaky ReLU.

**Numerical issues.** Comprobar:
- Input normalization: media ~0, std ~1.
- Pesos sin NaN o inf.
- Loss sin NaN o inf (usa `torch.autograd.detect_anomaly`).
- Gradient norm < 1e6 y > 1e-8.

**Cuándo pedir ayuda vs seguir debuggeando.** Si después de
1 hora de debugging con este checklist no avanzas, es
señal de que el problema es fundamental (conceptual, no
técnico). Vuelve a leer el paper, busca issues en GitHub, o
pregunta a la comunidad.

## Constrúyelo

```python
import numpy as np


class DebugMonitor:
    """Monitor ligero para diagnosticar entrenamiento."""

    def __init__(self):
        self.history = {"loss": [], "grad_norm": [],
                        "lr": [], "dead_neurons": []}

    def log_step(self, loss, grads, lr, activations=None):
        self.history["loss"].append(float(loss))
        if grads is not None:
            norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))
            self.history["grad_norm"].append(norm)
        self.history["lr"].append(float(lr))
        if activations is not None:
            # Detectar neuronas muertas: activación media < 0.01
            # en una capa con ReLU
            for i, a in enumerate(activations):
                if a.size > 0:
                    dead = float(np.mean(np.abs(a) < 0.01))
                    self.history["dead_neurons"].append(
                        {"layer": i, "fraction": dead}
                    )

    def check_gradient_health(self):
        norms = self.history["grad_norm"]
        if not norms:
            return "Sin gradientes registrados"
        last = norms[-1]
        if last > 1e5:
            return f"ALERTA: grad_norm = {last:.2e}, posible explosión"
        if last < 1e-8:
            return f"ALERTA: grad_norm = {last:.2e}, posible vanishing"
        return f"OK: grad_norm = {last:.2e}"

    def check_loss_trajectory(self):
        losses = self.history["loss"]
        if len(losses) < 2:
            return "Sin suficientes losses"
        delta = losses[-1] - losses[0]
        if delta > 0:
            return f"ALERTA: loss subió de {losses[0]:.3f} a {losses[-1]:.3f}"
        if delta > -0.01 * losses[0]:
            return f"AVISO: loss apenas cambió ({delta:.3f})"
        return f"OK: loss bajó de {losses[0]:.3f} a {losses[-1]:.3f}"


def gradient_clip(grads, max_norm=1.0):
    """Gradient clipping por norma total."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in grads]
    return grads


def overfit_test(model_fn, X_batch, y_batch, n_steps=100, lr=0.01):
    """Test pre-flight: el modelo puede memorizar 32 ejemplos?"""
    params = model_fn.init_params()
    for _ in range(n_steps):
        loss, grads = model_fn.loss_and_grad(params, X_batch, y_batch)
        params = model_fn.update(params, grads, lr)
    return loss
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-debug-nn
fase: 03
leccion: 13
---

Eres un asistente que diagnostica problemas en el entrena-
miento de redes neuronales. Recibirás la loss curve, las
normas de gradiente, y los síntomas. Tu trabajo:

1. Si loss no baja: primero overfit a un mini-batch.
2. Si gradientes son 0: cambiar ReLU a leaky ReLU.
3. Si gradientes explotan: gradient clipping a 1.0.
4. Si loss oscila: reducir lr 10x.
5. Si val loss sube mientras train baja: regularización.
6. Recomienda siempre TensorBoard o W&B.
7. Sugerir learning rate finder.
8. Si la loss es NaN desde el inicio: verificar input
   pipeline.
9. Si después de 1 hora no avanza: revisar el paper o
   issues en GitHub.
```

## Ejercicios

1. **Overfit test**: implementa un MLP y verifica que puede
   memorizar un mini-batch de 32 ejemplos.
2. **Gradient norm monitor**: añade logging de la norma de
   gradientes por capa en el entrenamiento de un MLP.
3. **Desafío**: reproduce el experimento del paper
   "Understanding the difficulty of training deep
   feedforward neural networks" (Glorot & Bengio, 2010).

## Lecturas recomendadas

- *A Recipe for Training Neural Networks* — Karpathy, 2019
  (blog post clásico).
- *Deep Learning* — Goodfellow et al. (cap. 8 sobre
  optimización, troubleshooting).
- PyTorch Debugging: <https://pytorch.org/docs/stable/debug.html>.
- W&B: <https://wandb.ai>.
- TensorBoard: <https://www.tensorflow.org/tensorboard>.

---

> 📚 **Adaptación al español** de la lección "[Debugging Neural Networks]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
