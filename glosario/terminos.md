# 📖 Glosario

> **Adaptación al español** del glosario de
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../CREDITS.md).

Este glosario contiene las definiciones canónicas de los términos
técnicos que aparecen a lo largo del currículo. Cuando introduzcas un
término nuevo usado por más de una lección, añádelo aquí.

Las definiciones están pensadas para **una persona que ya programa pero
está empezando en IA**. Si necesitas una explicación más profunda, sigue
los enlaces a la lección correspondiente.

---

## A

### Agent (agente)
Programa que percibe su entorno, decide qué herramienta invocar y ejecuta
acciones hasta cumplir un objetivo. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Attention (atención)
Mecanismo que permite a un modelo ponderar dinámicamente la importancia de
cada token de entrada al producir una salida. Núcleo de los Transformers.
Ver [fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Adam / AdamW
Optimizador estocástico que combina momentum y tasas de aprendizaje
adaptativas. Variante `AdamW` desacopla la decaimiento de pesos de la
actualización del gradiente. Ver [fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## B

### Backpropagation (propagación hacia atrás)
Algoritmo para calcular gradientes de la pérdida respecto a los pesos
de una red neuronal, aplicando la regla de la cadena en orden inverso.
Ver [fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Bias (sesgo, en ML)
- **Sesgo estadístico:** diferencia sistemática entre la estimación de
  un modelo y el valor verdadero.
- **Sesgo del modelo:** término independiente en una regresión lineal
  (`b` en `y = w·x + b`).
- **Sesgo social:** prejuicios presentes en datos o algoritmos. Ver
  [fase 18 — Ética y alineación](../fases/18-etica-y-alineacion/README.md).

---

## C

### Checkpoint
Instantánea de los pesos de un modelo en un punto del entrenamiento. Se
usa para reanudar el entrenamiento o para inferencia. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Cross-entropy (entropía cruzada)
Función de pérdida usada en clasificación. Mide la divergencia entre la
distribución predicha y la verdadera. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## D

### Dataset
Conjunto de ejemplos `(entrada, salida)` usado para entrenar, validar o
evaluar un modelo.

### Dropout
Técnica de regularización que "apaga" aleatoriamente un porcentaje de
neuronas en cada paso de entrenamiento para evitar sobreajuste. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## E

### Embedding (vector de incrustación)
Representación densa y de dimensión fija de un símbolo (palabra, imagen,
usuario) aprendida por el modelo. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Epoch (época)
Una pasada completa del conjunto de entrenamiento durante el
entrenamiento de un modelo.

---

## F

### Fine-tuning (ajuste fino)
Re-entrenamiento de un modelo pre-entrenado con datos específicos de
una tarea para especializarlo. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Function calling
Capacidad de un LLM para emitir llamadas estructuradas a funciones
externas en lugar de (o además de) texto. Ver
[fase 13 — Herramientas y protocolos](../fases/13-herramientas-y-protocolos/README.md).

---

## G

### Gradient descent (descenso por gradiente)
Algoritmo de optimización iterativo que ajusta los pesos en la dirección
opuesta al gradiente de la pérdida. Variantes: SGD, Momentum, Adam. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Grounding (anclaje)
Práctica de fundamentar las respuestas de un LLM en evidencia externa
(documentos, búsqueda, herramientas) para reducir alucinaciones. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

---

## H

### Hallucination (alucinación)
Respuesta de un modelo que parece coherente pero no se sostiene en los
datos de entrada ni en conocimiento verificable.

---

## I

### Inference (inferencia)
Proceso de generar predicciones con un modelo ya entrenado. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

---

## L

### LLM (Large Language Model)
Modelo de lenguaje de gran tamaño entrenado con objetivos de
auto-regresión o auto-codificación sobre corpus masivos. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### LoRA (Low-Rank Adaptation)
Técnica de fine-tuning eficiente en parámetros que adapta un modelo
agregando matrices de bajo rango. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Loss (pérdida)
Función escalar que mide qué tan mal lo está haciendo el modelo. El
entrenamiento busca minimizarla. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## M

### MCP (Model Context Protocol)
Protocolo abierto para que los LLMs invoquen herramientas externas de
manera estandarizada. Ver
[fase 13 — Herramientas y protocolos](../fases/13-herramientas-y-protocolos/README.md).

### MLOps
Conjunto de prácticas para desplegar, monitorear y mantener modelos de
ML en producción. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### MLOps loop
Ciclo continuo de entrenamiento → despliegue → monitoreo → re-entrenamiento.

---

## N

### NLP (Natural Language Processing)
Subcampo de la IA dedicado a procesar y generar lenguaje humano. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Normalization (normalización)
Re-escalado de features para tener media 0 y desviación 1 (u otra
convención) y estabilizar el entrenamiento.

---

## O

### Overfitting (sobreajuste)
Fenómeno en el que el modelo memoriza los datos de entrenamiento y
generaliza mal a datos nuevos. Se mitiga con regularización, más datos
o early stopping.

### Optimization (optimización)
Proceso de ajustar los parámetros de un modelo para minimizar la
pérdida.

---

## P

### Prompt
Texto de entrada que se envía a un modelo de lenguaje. La ingeniería de
prompts estudia cómo escribirlos para obtener mejores resultados. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Prompt injection (inyección de prompt)
Ataque en el que un usuario malicioso introduce instrucciones en el
contexto del modelo para que ignore las instrucciones originales. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Perceptron
Modelo matemático de una neurona artificial: combinación lineal
seguida de una función de activación. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## R

### RAG (Retrieval-Augmented Generation)
Patrón en el que un LLM recupera documentos relevantes antes de generar
una respuesta, reduciendo alucinaciones. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Regularization (regularización)
Técnica para penalizar modelos complejos y reducir el sobreajuste
(L1, L2, dropout, data augmentation).

### ReLU (Rectified Linear Unit)
Función de activación `max(0, x)`. Es la más usada en redes profundas
por su simplicidad y por evitar el problema del gradienteevanescente.

### RLHF (Reinforcement Learning from Human Feedback)
Técnica de alineación que ajusta un modelo usando preferencias humanas
codificadas en un modelo de recompensa. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Reward model (modelo de recompensa)
Modelo que predice qué tan buena es una salida según las preferencias
humanas; núcleo del RLHF.

---

## S

### Softmax
Función que convierte un vector de puntajes en una distribución de
probabilidad. Usada en la capa de salida de clasificadores.

### Stochastic gradient descent (SGD)
Variante del descenso por gradiente que usa un subconjunto aleatorio
(mini-batch) de los datos en cada paso.

### Supervised learning (aprendizaje supervisado)
Paradigma en el que el modelo aprende de pares `(entrada, etiqueta)`.
Ver [fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Self-supervised learning
Paradigma en el que la "etiqueta" se deriva de los propios datos
(rotar una imagen, predecir la siguiente palabra, etc.).

---

## T

### Token
Unidad mínima que procesa un LLM. Puede ser una palabra, sub-palabra o
carácter. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Tokenizer
Componente que convierte texto crudo en una secuencia de tokens. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Transformer
Arquitectura de red neuronal basada exclusivamente en mecanismos de
atención. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Transfer learning (aprendizaje por transferencia)
Reutilización de un modelo pre-entrenado en una tarea como punto de
partida para otra tarea relacionada.

### Training (entrenamiento)
Proceso iterativo de ajustar los pesos de un modelo para minimizar la
pérdida en un conjunto de datos.

---

## U

### Unsupervised learning (aprendizaje no supervisado)
Paradigma en el que el modelo aprende la estructura de los datos sin
etiquetas. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

---

## V

### Validation set (conjunto de validación)
Subconjunto de los datos separado del entrenamiento, usado para
ajustar hiperparámetros.

### Variance (varianza)
Sensibilidad del modelo a fluctuaciones en los datos de entrenamiento.
Varianza alta ⇒ sobreajuste.

---

## W

### Weight (peso)
Parámetro aprendible de un modelo. Una red neuronal tiene miles o
millones de pesos.

### Word embedding (incrustación de palabras)
Vector denso que representa una palabra, aprendido de forma que
palabras similares tengan vectores cercanos. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

---

## Y

### YAML
Formato de serialización legible por humanos usado para configuraciones
y frontmatter de lecciones y skills.

---

## Z

### Zero-shot (sin ejemplos)
Capacidad de un modelo de realizar una tarea sin haber visto ejemplos
de esa tarea durante el entrenamiento.

---

## Cómo añadir un término

1. Edita este archivo.
2. Ubica el término en orden alfabético bajo la letra correspondiente.
3. Mantén la definición breve (3-6 líneas) y enlaza a la fase o lección
   relevante usando rutas relativas.
4. No añadas términos cuyo uso esté limitado a **una sola lección**;
   defínelos en el `docs/es.md` de esa lección.

---

> **Aviso:** los términos de este glosario son **adaptaciones** al español
> del glosario original en inglés de AI Engineering from Scratch
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../CREDITS.md).
