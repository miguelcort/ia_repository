# Fase 3 — Núcleo de Deep Learning

> Redes neuronales desde los primeros principios. Sin frameworks hasta haber construido uno.

Esta fase es la **transición de ML clásico a deep learning**. La idea
es derivar y programar, capa a capa, los componentes que las
bibliotecas como PyTorch o JAX nos abstraen: el perceptrón, la
propagación hacia adelante, la propagación hacia atrás, las funciones
de activación, las pérdidas, los optimizadores, la regularización, la
inicialización y los *schedules* de learning rate. Cuando lleguemos a
PyTorch en la lección 11, el estudiante ya no verá "magia" detrás de
`loss.backward()`: sabrá exactamente qué gradientes se están
calculando y por qué la regla de la cadena es la columna vertebral de
todo el sistema.

La fase sigue un orden pedagógico estricto. Primero, **una neurona**
(lección 01). Después, **una red de varias capas** (lección 02).
Luego, **backprop** desde cero (lección 03). A partir de ahí
agregamos los "trucos del oficio" que permiten entrenar redes
profundas: activaciones (04), pérdidas (05), optimizadores (06),
regularización (07), inicialización (08), schedules (09). La
lección 10 **integra todo en un mini-framework** propio de ~200
líneas. Las últimas tres lecciones (11, 12, 13) son el **puente a
los frameworks de producción**: el estudiante repite la misma tarea
en PyTorch y en JAX, y aprende las técnicas de depuración que
separan un entrenamiento exitoso de uno que se queda atascado
divergiendo.

## Índice de lecciones

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [El perceptrón: donde todo empezó](01-el-perceptron/) | Construir | Neurona de McCulloch-Pitts y perceptrón de Rosenblatt. |
| 02 | [Redes multi-capa y forward pass](02-redes-multicapa/) | Construir | Composición de capas, activaciones y forward en NumPy. |
| 03 | [Backpropagation desde cero](03-backpropagation/) | Construir | Regla de la cadena aplicada a un MLP de dos capas. |
| 04 | [Funciones de activación: ReLU, Sigmoid, GELU](04-funciones-de-activacion/) | Construir | Cuándo cada una y por qué ReLU aceleró el deep learning. |
| 05 | [Funciones de pérdida: MSE, entropía cruzada, contrastiva](05-funciones-de-perdida/) | Construir | Derivadas, propiedades y elección por tarea. |
| 06 | [Optimizadores: SGD, Momentum, Adam, AdamW](06-optimizadores/) | Construir | De SGD a AdamW con código comparativo. |
| 07 | [Regularización: dropout, weight decay, batch norm](07-regularizacion/) | Construir | Cómo evitar el sobreajuste más allá de la early stopping. |
| 08 | [Inicialización de pesos y estabilidad](08-inicializacion-de-pesos/) | Construir | Xavier, He, LeCun y el problema de vanishing/exploding. |
| 09 | [Schedules de learning rate y warmup](09-programacion-de-learning-rate/) | Construir | Step, cosine, warmup, OneCycle. |
| 10 | [Construye tu propio mini-framework](10-mini-framework/) | Construir | Autograd, módulos, entrenamiento y serialización. |
| 11 | [Introducción a PyTorch](11-introduccion-a-pytorch/) | Construir | Tensores, `nn.Module`, `optim`, `DataLoader`. |
| 12 | [Introducción a JAX](12-introduccion-a-jax/) | Construir | `grad`, `jit`, `vmap`, `pmap` y comparativa con PyTorch. |
| 13 | [Depuración de redes neuronales](13-depuracion-de-redes-neuronales/) | Construir | Detección de NaN, gradientes, *gradient checkpointing*. |

## Prerrequisitos

- **Fase 0** completa (entorno).
- **Fase 1** completa (álgebra lineal, cálculo, gradiente) — la
  comprensión de backprop depende de entender derivada parcial y
  regla de la cadena.
- **Fase 2** completa o equivalente (regresión logística, árboles).
- NumPy fluido.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Derivar** la regla de actualización de pesos de un MLP de dos
  capas y verificarla con un *checkgrad* numérico.
- **Construir** un autograd mínimo con *reverse-mode* automático
  (mini-framework de la lección 10).
- **Elegir** función de activación, optimizador y schedule
  adecuados para una tarea dada, justificando la elección.
- **Diagnosticar** vanishing/exploding gradients, NaN en la
  pérdida, y *dead ReLUs* con técnicas de inspección.
- **Migrar** el mismo modelo a PyTorch y a JAX sin reescribir la
  lógica, entendiendo las diferencias de paradigma.
- **Depurar** entrenamientos divergentes con `torch.autograd.detect_anomaly`,
  `tensorboard`, *gradient clipping* y visualización de
  activaciones.

## Stack y herramientas

- **NumPy** para las primeras diez lecciones.
- **PyTorch** (≥ 2.0) desde la lección 11.
- **JAX** (con `jaxlib` y `optax`) en la lección 12.
- **Matplotlib** para visualizar pérdidas y activaciones.
- **TensorBoard** o `torch.utils.tensorboard` opcional.
- **pytest** (a través de `unittest`) para las pruebas unitarias.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Perceptrón** | Lección 01 | Contexto histórico, base de las redes. |
| **Backprop** | Lección 03 | Toda red neuronal del currículo. |
| **ReLU** | Lección 04 | Fase 4 (CNN), Fase 7 (transformers). |
| **Cross-entropy** | Lección 05 | Fase 5 (clasificación de texto), Fase 10 (LLM). |
| **AdamW** | Lección 06 | Fase 10 (entrenamiento de LLM), Fase 11 (fine-tuning). |
| **Dropout** | Lección 07 | Fase 4, Fase 7, regularización moderna. |
| **Inicialización He** | Lección 08 | Redes profundas en todas las fases. |
| **Cosine schedule** | Lección 09 | Fase 10 (entrenamiento de LLM). |
| **Autograd** | Lecciones 10, 11 | Base de PyTorch, JAX y TensorFlow. |
| **Gradient checkpointing** | Lección 13 | Fase 10 (entrenamiento de LLMs grandes). |

## Cómo estudiar esta fase

1. **Programa primero en NumPy, luego en PyTorch.** La tentación de
   saltar a PyTorch es enorme; resístela hasta terminar la lección
   10. Sin la base en NumPy, los errores de PyTorch parecen
   *misteriosos* en lugar de *matemáticos*.
2. **Compara siempre con un *checkgrad* numérico.** En la lección
   03 implementas una verificación con diferencias finitas; úsala
   como red de seguridad cada vez que escribas una nueva capa.
3. **Grafica la pérdida y la norma del gradiente** desde la primera
   lección de entrenamiento. Sin esa gráfica, no sabes si la red
   está aprendiendo o divergiendo.
4. **La lección 10 es el examen real.** Si puedes escribir un MLP
   en el mini-framework y entrenarlo en MNIST, dominas la fase.
5. **En PyTorch y JAX, reescribe el mismo ejemplo de la lección
   10.** La repetición de la misma tarea con tres *stacks* (NumPy,
   PyTorch, JAX) consolida los conceptos.

## Verificación de progreso

```bash
# Lección 03 — backprop con checkgrad
python3 fases/03-nucleo-deep-learning/03-backpropagation/code/main.py

# Lección 10 — mini-framework entrenando MNIST
python3 fases/03-nucleo-deep-learning/10-mini-framework/code/main.py

# Lección 11 — misma red en PyTorch, mismo accuracy
python3 fases/03-nucleo-deep-learning/11-introduccion-a-pytorch/code/main.py
```

Si el mini-framework y la versión PyTorch convergen al mismo
accuracy en MNIST (±1%), la fase está aprobada.

## Conexión con otras fases

- **Entrada** → [Fase 2 — Fundamentos de ML](../02-fundamentos-ml/README.md).
- **Salida natural** → [Fase 4 — Visión por computador](../04-vision-por-computador/README.md),
  donde las redes convolucionales son el siguiente nivel de
  arquitectura.
- **Reuso en** → Fase 5 (embeddings), Fase 7 (transformers),
  Fase 9 (policy gradient con PyTorch), Fase 10 (entrenamiento
  de LLM), Fase 11 (fine-tuning).

## Recursos recomendados

- *Deep Learning* — Goodfellow, Bengio, Courville (libre en línea).
- *Neural Networks and Deep Learning* — Michael Nielsen (libre).
- *Dive into Deep Learning* — Zhang, Lipton, Smola (libre).
- *PyTorch tutorials* — <https://pytorch.org/tutorials>.
- *JAX documentation* — <https://jax.readthedocs.io>.
- *3Blue1Brown — But what is a neural network?* (YouTube).
- *Fast.ai — Practical Deep Learning for Coders*.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *backprop*,
  *dropout*, *Adam*, *weight decay*, *batch norm*.
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.
- [PLANTILLA_LECCION.md](../../PLANTILLA_LECCION.md) — estructura
  de cada lección.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
