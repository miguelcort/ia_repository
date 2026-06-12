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

### Agentic loop (loop agéntico)
Ciclo perceive → plan → act → observe que ejecutan los agentes. Es la
abstracción fundamental sobre la que se construyen todos los
frameworks (LangGraph, AutoGen, CrewAI). Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### A2A (Agent-to-Agent)
Protocolo de Google para que agentes de distintos proveedores se
descubran e invoquen entre sí mediante agent cards y JSON-RPC. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

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
- **Sesgo social:** prejucio presentes en datos o algoritmos. Ver
  [fase 18 — Ética y alineación](../fases/18-etica-y-alineacion/README.md).

### Blackwell (arquitectura)
Generación de GPUs de NVIDIA posterior a Hopper (H100). Ofrece FP4
nativo, 192 GB de memoria y hasta 4500 TFLOPS en FP8. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## C

### Checkpoint
Instantánea de los pesos de un modelo en un punto del entrenamiento. Se
usa para reanudar el entrenamiento o para inferencia. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Constitutional AI (IA constitucional)
Método de alineación de Anthropic en el que un LLM se crítica y
revisa a sí mismo frente a un conjunto de principios explícitos
(constitución) en lugar de depender únicamente de RLHF. Ver
[fase 15 — Sistemas autónomos](../fases/15-sistemas-autonomos/README.md).

### Continuous batching
Técnica de serving que inserta nuevas solicitudes en un lote de
inferencia en marcha, manteniendo la GPU saturada y mejorando el
throughput. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Cross-entropy (entropía cruzada)
Función de pérdida usada en clasificación. Mide la divergencia entre la
distribución predicha y la verdadera. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Costo marginal por token
Precio real que un proveedor cobra por cada token adicional (input o
output), excluyendo el costo fijo de mantener el servicio encendido.
Base de los modelos de pricing por inferencia. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## D

### Dataset
Conjunto de ejemplos `(entrada, salida)` usado para entrenar, validar o
evaluar un modelo.

### Disaggregated prefill-decode
Patrón de serving que separa la fase de prefill (compute-bound) de la
fase de decode (memory-bound) en GPUs distintas, mejorando el
throughput agregado. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Dropout
Técnica de regularización que "apaga" aleatoriamente un porcentaje de
neuronas en cada paso de entrenamiento para evitar sobreajuste. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## E

### Edge inference
Inferencia ejecutada en el dispositivo del usuario final (móvil, laptop,
dispositivo IoT) en lugar de en un servidor cloud. Reduce latencia y
preserva privacidad, pero limita el tamaño del modelo. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Embedding (vector de incrustación)
Representación densa y de dimensión fija de un símbolo (palabra, imagen,
usuario) aprendida por el modelo. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Epoch (época)
Una pasada completa del conjunto de entrenamiento durante el
entrenamiento de un modelo.

### Error budget (presupuesto de error)
Cantidad máxima de fallos permitida en un periodo para un SLO
determinado. Cuando se consume, los equipos deben pausar despliegues
de riesgo. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## F

### FIPA ACL
Lenguaje estándar (FIPA Foundation) para mensajes entre agentes
inteligentes. Define performativas como `inform`, `request`, `propose`
y `cfp`. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

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

### Goodput (throughput útil)
Fracción de requests que cumplen el SLO de latencia establecido, en
oposición al throughput bruto. Es la métrica de salud real de un
servicio. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### Handoff
Transferencia explícita de control de un agente a otro, típicamente con
contexto compartido. Patrón nativo del OpenAI Agents SDK. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

---

## I

### Inference (inferencia)
Proceso de generar predicciones con un modelo ya entrenado. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

---

## K

### KV cache
Memoria intermedia que almacena las claves y valores de atención
precalculadas para no recalcularlos en cada token generado. Su tamaño
es proporcional a la longitud del contexto y al número de
solicitudes concurrentes. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## L

### LLM (Large Language Model)
Modelo de lenguaje de gran tamanho entrenado con objetivos de
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

### MAST (Multi-Agent System Failures)
Taxonomía de Anthropic que clasifica los fallos de sistemas
multi-agente en 14 categorías (verificación, descomposición, prompt
injection, groupthink, free-riding, etc.). Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

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

### Model routing
Estrategia que enruta cada request al modelo más apropiado según
complejidad, costo o latencia objetivo. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### OpenTelemetry (OTel)
Estándar abierto para tracing, métricas y logs distribuidos, con
SDKs en varios lenguajes y exportadores a backends como Jaeger o
Honeycomb. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## P

### PagedAttention
Algoritmo de gestión de memoria de vLLM que divide la KV cache en
bloques de tamaño fijo (como la paginación del SO) para eliminar
fragmentación y aumentar el throughput. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Plan-and-execute
Patrón de agente en el que primero se genera un plan completo y
luego se ejecuta paso a paso, con re-planning opcional si un paso
falla. ReWOO es un ejemplo prominente. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Prefix cache
Caché que reutiliza la KV cache del prefijo común entre solicitudes,
evitando recomputación. Implementado en vLLM (Automatic Prefix
Caching) y SGLang (RadixAttention). Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Progressive rollout
Estrategia de despliegue que expone una nueva versión a un porcentaje
creciente de tráfico (1% → 10% → 50% → 100%), comparando métricas
contra la versión estable antes de avanzar. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Prompt
Texto de entrada que se envía a un modelo de lenguaje. La ingeniería de
prompts estudia cómo escribirlos para obtener mejores resultados. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Prompt injection (inyección de prompt)
Ataque en el que un usuario malicioso introduce instrucciones en el
contexto del modelo para que ignore las instrucciones originales. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Perceptron
Modelo matemático de una neurona artificial: combination lineal
seguida de una función de activación. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## Q

### Quantization (cuantización)
Conversión de los pesos y activaciones de un modelo de precisión
float (FP16/BF16) a precisión más baja (INT8, INT4, FP8, FP4) para
reducir memoria y acelerar inferencia, a costa de algo de accuracy.
Ver [fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Queue (cola)
Estructura de datos FIFO usada como buffer entre productores y
consumidores. En serving de LLMs actúa como backpressure cuando la
demanda supera la capacidad de inferencia. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### RLAIF (RL from AI Feedback)
Variante de RLHF donde el feedback lo genera otro LLM (un "LLM juez")
en lugar de humanos, escalando el proceso de alineación. Ver
[fase 15 — Sistemas autónomos](../fases/15-sistemas-autonomos/README.md).

### Round-robin
Estrategia de selección determinista en la que los agentes se turnan
para hablar en un orden fijo. Simple y justa, sin diversidad. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

---

## S

### Semantic cache
Caché de respuestas de LLM indexado por la similitud semántica del
prompt (usando embeddings), en lugar de una clave exacta. Reduce
costos y latencia en prompts repetitivos. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Shadow deployment
Estrategia de despliegue donde la nueva versión recibe tráfico real
en paralelo pero sin devolver sus respuestas a los usuarios;
permite comparar el comportamiento contra la versión estable. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### SLO (Service Level Objective)
Objetivo medible de fiabilidad o rendimiento (e.g., 99.9% de
disponibilidad, p95 de latencia < 200 ms). Diferente de SLA, que
es la consecuencia contractual. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### SLI (Service Level Indicator)
Métrica cuantitativa que se observa para medir un SLO (e.g.,
`requests_successful / total_requests`). Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Softmax
Función que convierte un vector de puntajes en una distribución de
probabilidad. Usada en la capa de salida de clasificadores.

### Stochastic gradient descent (SGD)
Variante del descenso por gradiente que usa un subconjunto aleatorio
(mini-batch) de los datos en cada paso.

### Speculative decoding
Técnica que usa un modelo borrador pequeño para generar varios tokens
candidatos y un modelo target más grande para verificarlos en
paralelo, acelerando la inferencia 2-3x sin pérdida de calidad. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Supervised learning (aprendizaje supervisado)
Paradigma en el que el modelo aprende de pares `(entrada, etiqueta)`.
Ver [fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Self-supervised learning
Paradigma en el que la "etiqueta" se deriva de los propios datos
(rotar una imagen, predecir la siguiente palabra, etc.).

### SRE (Site Reliability Engineering)
Disciplina de Google que combina desarrollo y operaciones para
producir software ultra-confiable, con prácticas como SLIs/SLOs,
error budgets, blameless postmortems y runbooks. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Swarm intelligence
Comportamiento colectivo emergente de agentes simples (PSO, ACO) que
resuelve problemas complejos sin control centralizado. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

---

## T

### TBT (Time Between Tokens) / TPOT (Time Per Output Token)
Latencia promedio entre tokens consecutivos generados por un LLM.
Complementa a TTFT para describir la experiencia de streaming. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Token
Unidad mínima que procesa un LLM. Puede ser una palabra, sub-palabra o
carácter. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Tokenizer
Componente que convierte texto crudo en una secuencia de tokens. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### TTFT (Time To First Token)
Latencia desde que se envía el prompt hasta que el LLM emite el primer
token. Métrica clave de "responsiveness" en serving. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### Throughput (caudal)
Número de unidades de trabajo procesadas por unidad de tiempo
(tokens/segundo, requests/segundo, etc.). Métrica base de capacidad
del sistema. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### vLLM
Sistema de serving open-source de alto rendimiento para LLMs, basado
en PagedAttention y continuous batching. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## W

### Weight (peso)
Parámetro aprendible de un modelo. Una red neuronal tiene miles o
millones de pesos.

### Word embedding (incrustación de palabras)
Vector denso que representa una palabra, aprendido de forma que
palabras similares tengan vectores cercanos. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Workbench (para agentes)
Conjunto mínimo viable de componentes (tool registry, memory, run
loop, plan, verifier, gates) que necesita un agente para ser
confiable en producción. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

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

### ZOPA (Zone of Possible Agreement)
En negociación entre agentes, el rango de ofertas que ambas partes
aceptarían. Si el ZOPA está vacío, no hay acuerdo posible. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

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
