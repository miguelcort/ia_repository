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

### Action space (espacio de acciones)
Conjunto de todas las acciones posibles que un agente puede tomar en
un entorno dado. En RL puede ser discreto o continuo. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Adam / AdamW
Optimizador estocástico que combina momentum y tasas de aprendizaje
adaptativas. Variante `AdamW` desacopla la decaimiento de pesos de la
actualización del gradiente. Ver [fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### A2A (Agent-to-Agent)
Protocolo de Google para que agentes de distintos proveedores se
descubran e invoquen entre sí mediante agent cards y JSON-RPC. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

### Agent (agente)
Programa que percibe su entorno, decide qué herramienta invocar y ejecuta
acciones hasta cumplir un objetivo. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Agentic loop (loop agéntico)
Ciclo perceive → plan → act → observe que ejecutan los agentes. Es la
abstracción fundamental sobre la que se construyen todos los
frameworks (LangGraph, AutoGen, CrewAI). Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Attention (atención)
Mecanismo que permite a un modelo ponderar dinámicamente la importancia de
cada token de entrada al producir una salida. Núcleo de los Transformers.
Ver [fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Autoencoder
Red neuronal que aprende a comprimir (encoder) y reconstruir
(decoder) sus entradas. Sirve como base para VAE, difusión y
representaciones. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

---

## B

### Backbone
Red neuronal grande pre-entrenada (e.g., ResNet, ViT, BERT) que se
reutiliza como extractor de features para tareas posteriores. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Backpropagation (propagación hacia atrás)
Algoritmo para calcular gradientes de la pérdida respecto a los pesos
de una red neuronal, aplicando la regla de la cadena en orden inverso.
Ver [fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Bayes (teorema)
Regla fundamental de probabilidad condicional: `P(A|B) = P(B|A) P(A) / P(B)`.
Es la base de razonamiento bajo incertidumbre, clasificadores
Naive Bayes, inferencia bayesiana y MCMC. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Bias (sesgo, en ML)
- **Sesgo estadístico:** diferencia sistemática entre la estimación de
  un modelo y el valor verdadero.
- **Sesgo del modelo:** término independiente en una regresión lineal
  (`b` en `y = w·x + b`).
- **Sesgo social:** prejuicio presentes en datos o algoritmos. Ver
  [fase 18 — Ética y alineación](../fases/18-etica-y-alineacion/README.md).

### Black (box) / White (box)
- **Caja negra:** modelo cuyo interior no se inspecciona; solo se
observa input → output. Los LLMs son cajas negras.
- **Caja blanca:** modelo interpretable (regresión lineal, árbol). Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Blackwell (arquitectura)
Generación de GPUs de NVIDIA posterior a Hopper (H100). Ofrece FP4
nativo, 192 GB de memoria y hasta 4500 TFLOPS en FP8. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### BLEU
Métrica clásica de evaluación de traducción automática y generación de
texto que mide la superposición de n-gramas con una o más referencias. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### BPE (Byte Pair Encoding)
Algoritmo de tokenización subpalabra que itera fusionando los pares
de bytes más frecuentes. Base de GPT, Llama, Mistral. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

---

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

### Calibration (calibración)
Propiedad de un clasificador: que sus probabilidades predichas
reflejen frecuencias reales. Un modelo bien calibrado asigna
`P=0.8` cuando tiene razón el 80% de las veces. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Checkpoint
Instantánea de los pesos de un modelo en un punto del entrenamiento. Se
usa para reanudar el entrenamiento o para inferencia. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Chain-of-thought (CoT, cadena de pensamiento)
Técnica de prompting que induce al LLM a razonar paso a paso antes de
producir la respuesta final. Mejora sustancialmente tareas de
lógica y matemáticas. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### CNN (Convolutional Neural Network)
Red neuronal con capas convolucionales que aplica filtros
deslizantes para extraer patrones espaciales. Base de la visión por
computador moderna. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Constitutional AI (IA constitucional)
Método de alineación de Anthropic en el que un LLM se crítica y
revisa a sí mismo frente a un conjunto de principios explícitos
(constitución) en lugar de depender únicamente de RLHF. Ver
[fase 15 — Sistemas autónomos](../fases/15-sistemas-autonomos/README.md).

### Context window (ventana de contexto)
Número máximo de tokens que un LLM puede procesar en un solo
prompt. Gemini 1.5 tiene 1M, Claude 200k, GPT-4o 128k. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Continuous batching
Técnica de serving que inserta nuevas solicitudes en un lote de
inferencia en marcha, manteniendo la GPU saturada y mejorando el
throughput. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Contrastive learning
Paradigma de aprendizaje auto-supervisado en el que el modelo
aprende a acercar ejemplos similares y alejar ejemplos diferentes
(InfoNCE, SimCLR, CLIP). Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Cross-entropy (entropía cruzada)
Función de pérdida usada en clasificación. Mide la divergencia entre la
distribución predicha y la verdadera. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Cross-validation (validación cruzada)
Técnica para estimar el rendimiento de generalización dividiendo los
datos en k particiones y rotando cuál se usa como validación. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Costo marginal por token
Precio real que un proveedor cobra por cada token adicional (input o
output), excluyendo el costo fijo de mantener el servicio encendido.
Base de los modelos de pricing por inferencia. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

---

## D

### DALL-E
Modelo de OpenAI (y otros) que genera imágenes a partir de texto
usando difusión latente condicionada por CLIP. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

### DDPG / DQN / PPO
Familia de algoritmos de RL: DQN (value-based, off-policy),
DDPG (actor-critic para acciones continuas), PPO (policy
optimization, on-policy, estable). Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Dataset
Conjunto de ejemplos `(entrada, salida)` usado para entrenar, validar o
evaluar un modelo.

### Decision boundary (frontera de decisión)
Hiperplano (o superficie) que separa las clases predichas por un
clasificador. En 2D es una línea; en nD, un hiperplano. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Diffusion model (modelo de difusión)
Modelo generativo que aprende a revertir un proceso de ruido
gaussiano. Base de Stable Diffusion, DALL-E, Imagen. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

### Dimensionality reduction
Técnica para reducir el número de features conservando la
estructura (PCA, t-SNE, UMAP, autoencoders). Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Disaggregated prefill-decode
Patrón de serving que separa la fase de prefill (compute-bound) de la
fase de decode (memory-bound) en GPUs distintas, mejorando el
throughput agregado. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Distillation (destilación)
Técnica de compresión en la que un modelo pequeño ("student")
aprende a imitar al grande ("teacher") para igualar parte de su
rendimiento con mucho menos cómputo. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Dropout
Técnica de regularización que "apaga" aleatoriamente un porcentaje de
neuronas en cada paso de entrenamiento para evitar sobreajuste. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## E

### Early stopping
Técnica de regularización que detiene el entrenamiento cuando la
métrica de validación deja de mejorar, evitando sobreajuste. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Edge inference
Inferencia ejecutada en el dispositivo del usuario final (móvil, laptop,
dispositivo IoT) en lugar de en un servidor cloud. Reduce latencia y
preserva privacidad, pero limita el tamaño del modelo. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Eigenvalue / Eigenvector (valor propio / vector propio)
Para una matriz `A`, un escalar `λ` y vector `v ≠ 0` tales que `A v = λ v`.
Son la base de PCA, SVD y la diagonalización. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Embedding (vector de incrustación)
Representación densa y de dimensión fija de un símbolo (palabra, imagen,
usuario) aprendida por el modelo. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Encoder-decoder
Arquitectura de dos etapas: encoder resume la entrada a un vector y
el decoder genera la salida. Base de traducción automática, T5, BART
y muchos modelos seq2seq. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Entropy (entropía)
Medida de incertidumbre de una distribución de probabilidad. En teoría
de la información, `H(p) = -Σ p log p`. En RL, incentiva
exploración. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Epoch (época)
Una pasada completa del conjunto de entrenamiento durante el
entrenamiento de un modelo.

### Error budget (presupuesto de error)
Cantidad máxima de fallos permitida en un periodo para un SLO
determinado. Cuando se consume, los equipos deben pausar despliegues
de riesgo. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Exploding gradient
Problema de entrenamiento en redes profundas donde los gradientes
crecen exponencialmente y desbordan la representación en coma
flotante. Se mitiga con clipping, normalización y warmup. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## F

### F1-score (F-score)
Media armónica de precisión y recall. `F1 = 2 * P * R / (P + R)`.
Útil cuando las clases están desbalanceadas. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Feature (característica)
Variable de entrada que un modelo usa para hacer predicciones. También
llamada "atributo" o "variable independiente". Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### FFCV / Five-fold CV
Validación cruzada en 5 particiones. Cada partición actúa una vez
como validación y las otras 4 como entrenamiento; se promedian los
5 resultados. Estándar en ML clásico. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### FIPA ACL
Lenguaje estándar (FIPA Foundation) para mensajes entre agentes
inteligentes. Define performativas como `inform`, `request`, `propose`
y `cfp`. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

### Fine-tuning (ajuste fino)
Re-entrenamiento de un modelo pre-entrenado con datos específicos de
una tarea para especializarlo. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Flash Attention
Implementación de atención que reduce el uso de memoria de O(n²) a
O(n) mediante tiling y recomputación, sin aproximar la matemática. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Foundation model (modelo base)
Modelo grande pre-entrenado en datos a escala de internet que sirve
como base para múltiples tareas mediante fine-tuning o prompting
(GPT, Llama, Claude, Gemini). Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Few-shot
Inferencia en la que el prompt contiene 1-N ejemplos de la tarea antes
de pedirle al LLM la respuesta final. Reduce la necesidad de
fine-tuning. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Function calling
Capacidad de un LLM para emitir llamadas estructuradas a funciones
externas en lugar de (o además de) texto. Ver
[fase 13 — Herramientas y protocolos](../fases/13-herramientas-y-protocolos/README.md).

---

## G

### GAN (Generative Adversarial Network)
Par de redes (generator + discriminator) que compiten. El generator
intenta producir muestras realistas; el discriminator intenta
distinguir reales de falsas. Base de StyleGAN, CycleGAN. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

### Gaussian distribution (distribución gaussiana / normal)
Distribución de probabilidad con forma de campana caracterizada por
media `μ` y varianza `σ²`. Aparece en casi todo: ruido en
difusión, inicialización de pesos, supuestos lineales. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Goodput (throughput útil)
Fracción de requests que cumplen el SLO de latencia establecido, en
oposición al throughput bruto. Es la métrica de salud real de un
servicio. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### GPU (Graphics Processing Unit)
Procesador originalmente para gráficos que se descubrió ideal para
operaciones matriciales paralelas. Es la base de todo el deep
learning moderno. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

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

### Hopfield network
Red neuronal recurrente clásica (1982) que almacena patrones como
mínimos de energía. Antecesora conceptual de los Transformers y la
memoria asociativa. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Hyperparameter (hiperparámetro)
Configuración del modelo o del entrenamiento elegida antes de
entrenar (no aprendida), e.g., learning rate, batch size, número de
capas. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

---

## I

### Image segmentation
Tarea de visión de asignar una clase a cada pixel de una imagen.
- **Semántica:** misma clase para todos los pixels del mismo objeto.
- **De instancia:** distingue entre objetos individuales.
- **Panóptica:** combinación de ambas. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Inference (inferencia)
Proceso de generar predicciones con un modelo ya entrenado. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Instruction tuning
Ajuste fino supervisado de un LLM pre-entrenado con pares
(prompt, respuesta ideal) para que siga instrucciones. Base de
ChatGPT, Llama-Chat, etc. Ver
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

### Latent space (espacio latente)
Espacio de representaciones de menor dimensión aprendido por el
modelo (e.g., el z de un VAE, los embeddings de un LLM). Captura
la estructura semántica de los datos. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

### Layer normalization
Técnica de normalización que normaliza por muestra y por capa (no
por batch). Estable en tamaños de batch variables; usada en
Transformers. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### LLM (Large Language Model)
Modelo de lenguaje de gran tamanho entrenado con objetivos de
auto-regresión o auto-codificación sobre corpus masivos. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Logits
Salida cruda (sin softmax) de la última capa de un clasificador o
generador. Aplicar softmax convierte logits en probabilidades. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### LoRA (Low-Rank Adaptation)
Técnica de fine-tuning eficiente en parámetros que adapta un modelo
agregando matrices de bajo rango. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### Loss (pérdida)
Función escalar que mide qué tan mal lo está haciendo el modelo. El
entrenamiento busca minimizarla. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### L1 / L2 regularization
Penalizaciones añadidas a la loss para reducir el sobreajuste.
- **L1:** `λ Σ |w|` (sparse).
- **L2 / weight decay:** `λ Σ w²`. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### LSTM (Long Short-Term Memory)
Tipo de RNN con compuertas (input, forget, output) que mitiga el
problema del gradiente evanescente y captura dependencias largas. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

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

### MDP (Markov Decision Process)
Formalismo `(S, A, P, R, γ)` que define un entorno de RL: estados,
acciones, transiciones, recompensa y factor de descuento. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### MFCC (Mel-Frequency Cepstral Coefficients)
Características clásicas de audio que imitan la percepción humana del
sonido. Base del ASR y TTS tradicionales. Ver
[fase 6 — Voz y audio](../fases/06-voz-y-audio/README.md).

### Mixture of Experts (MoE)
Arquitectura donde solo una fracción de los parámetros ("experts")
se activa por token, permitiendo modelos enormes con cómputo
constante. Mixtral, GPT-4 rumored. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

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

### Multi-modal
Modelo o sistema que procesa más de una modalidad (texto, imagen,
audio, video). Ver
[fase 12 — IA multimodal](../fases/12-ia-multimodal/README.md).

### Multitask learning
Entrenar un modelo en varias tareas simultáneamente con la esperanza
de que el conocimiento compartido mejore cada tarea. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

---

## N

### Named Entity Recognition (NER)
Tarea de NLP de localizar y clasificar entidades mencionadas
(personas, organizaciones, lugares, fechas) en texto. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### N-gram
Secuencia contigua de N tokens. Usada en modelos de lenguaje
clásicos (unigrama, bigrama, trigrama) y en métricas como BLEU. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### NLP (Natural Language Processing)
Subcampo de la IA dedicado a procesar y generar lenguaje humano. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Normalization (normalización)
Re-escalado de features para tener media 0 y desviación 1 (u otra
convención) y estabilizar el entrenamiento.

### NumPy
Librería fundamental de Python para cómputo numérico con arrays
n-dimensionales y operaciones vectorizadas. Base de todo el
ecosistema de ML en Python. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

---

## O

### Object detection
Tarea de visión de localizar y clasificar múltiples objetos en una
imagen con bounding boxes (YOLO, Faster R-CNN, DETR). Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### OCR (Optical Character Recognition)
Tarea de extraer texto de imágenes. Tesseract es clásico; modelos
como TrOCR, Donut son modernos. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Off-policy / On-policy
- **On-policy:** el agente aprende de la misma política que ejecuta
  (PPO, A2C).
- **Off-policy:** aprende de datos generados por otra política
  (DQN, SAC). Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

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

### PCA (Principal Component Analysis)
Transformación lineal que proyecta los datos a un espacio de menor
dimensión maximizando la varianza preservada. Base de SVD y
muchas técnicas de reducción. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Plan-and-execute
Patrón de agente en el que primero se genera un plan completo y
luego se ejecuta paso a paso, con re-planning opcional si un paso
falla. ReWOO es un ejemplo prominente. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Perceptron
Modelo matemático de una neurona artificial: combination lineal
seguida de una función de activación. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Policy (política)
En RL, la estrategia `π(a|s)` que mapea estados a acciones. Puede
ser determinista o estocástica. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Positional encoding
Mecanismo para inyectar información de orden en una arquitectura
basada en atención (que es invariante al orden por sí misma). Puede
ser sinusoidal, aprendido o relativo (RoPE, ALiBi). Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Precision (precisión) / Recall (exhaustividad)
- **Precisión:** de los items predichos como positivos, cuántos son
  realmente positivos.
- **Recall:** de los items realmente positivos, cuántos fueron
  predichos como positivos. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

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

### PyTorch
Framework de deep learning con grafos dinámicos, base de la mayor
parte de la investigación y producción moderna. Ver
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

### ReAct (Reason + Act)
Patrón de agente que alterna pasos de razonamiento ("pensamiento")
con acciones sobre herramientas, mostrándolos explícitamente en el
prompt. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

### Recall (exhaustividad)
Ver **Precision**.

### Regularization (regularización)
Técnica para penalizar modelos complejos y reducir el sobreajuste
(L1, L2, dropout, data augmentation).

### ReLU (Rectified Linear Unit)
Función de activación `max(0, x)`. Es la más usada en redes profundas
por su simplicidad y por evitar el problema del gradienteevanescente.

### Representation learning
Aprendizaje de features útiles automáticamente a partir de datos
crudos (e.g., embeddings, autoencoders). Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### ResNet (Residual Network)
Arquitectura de CNN que introduce conexiones residuales (`y = F(x) + x`),
permitiendo entrenar redes de cientos o miles de capas. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### Reward (recompensa)
En RL, señal escalar que indica qué tan buena fue una acción en un
estado. Optimizar la suma acumulada de recompensas es el objetivo. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### RLHF (Reinforcement Learning from Human Feedback)
Técnica de alineación que ajusta un modelo usando preferencias humanas
codificadas en un modelo de recompensa. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Reward model (modelo de recompensa)
Modelo que predice qué tan buena es una salida según las preferencias
humanas; núcleo del RLHF.

### RNN (Recurrent Neural Network)
Red neuronal que procesa secuencias manteniendo un estado oculto
que se actualiza con cada token (LSTM, GRU). Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### RLAIF (RL from AI Feedback)
Variante de RLHF donde el feedback lo genera otro LLM (un "LLM juez")
en lugar de humanos, escalando el proceso de alineación. Ver
[fase 15 — Sistemas autónomos](../fases/15-sistemas-autonomos/README.md).

### RoPE (Rotary Position Embedding)
Codificación posicional que rota los vectores Q y K por ángulos
proporcionales a la posición, permitiendo extrapolación de longitud. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### Round-robin
Estrategia de selección determinista en la que los agentes se turnan
para hablar en un orden fijo. Simple y justa, sin diversidad. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

---

## S

### Sampling (muestreo)
Selección de un subconjunto de datos o de un subconjunto de la
distribución de salida. Incluye técnicas como top-k, top-p (nucleus)
y temperature scaling. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Scalability (escalabilidad)
Capacidad de un sistema de mantener su rendimiento al aumentar
el tamaño de la entrada, los datos, o el número de usuarios
concurrentes. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Self-supervised learning
Paradigma en el que la "etiqueta" se deriva de los propios datos
(rotar una imagen, predecir la siguiente palabra, etc.).

### Semantic cache
Caché de respuestas de LLM indexado por la similitud semántica del
prompt (usando embeddings), en lugar de una clave exacta. Reduce
costos y latencia en prompts repetitivos. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Semantic segmentation
Tarea de visión de asignar una clase a cada pixel de una imagen
(UNet, DeepLab, SegFormer). Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

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

### Speculative decoding
Técnica que usa un modelo borrador pequeño para generar varios tokens
candidatos y un modelo target más grande para verificarlos en
paralelo, acelerando la inferencia 2-3x sin pérdida de calidad. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### SRE (Site Reliability Engineering)
Disciplina de Google que combina desarrollo y operaciones para
producir software ultra-confiable, con prácticas como SLIs/SLOs,
error budgets, blameless postmortems y runbooks. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### State (estado)
En RL, una representación del entorno que el agente usa para
decidir la siguiente acción. El estado de Markov contiene toda la
información relevante. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Stochastic gradient descent (SGD)
Variante del descenso por gradiente que usa un subconjunto aleatorio
(mini-batch) de los datos en cada paso.

### Supervised learning (aprendizaje supervisado)
Paradigma en el que el modelo aprende de pares `(entrada, etiqueta)`.
Ver [fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Support Vector Machine (SVM)
Clasificador que encuentra el hiperplano de máximo margen entre
dos clases. Con kernel trick se vuelve no lineal. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### Swarm inteligencia
Comportamiento colectivo emergente de agentes simples (PSO, ACO) que
resuelve problemas complejos sin control centralizado. Ver
[fase 16 — Multi-agente y enjambres](../fases/16-multi-agente-y-enjambres/README.md).

### SVD (Singular Value Decomposition)
Factorización de una matriz `A = U Σ V^T` con `U` y `V` ortogonales
y `Σ` diagonal. Usada en PCA, LSA, compresión, recomendación. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

---

## T

### TBT (Time Between Tokens) / TPOT (Time Per Output Token)
Latencia promedio entre tokens consecutivos generados por un LLM.
Complementa a TTFT para describir la experiencia de streaming. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### TD(λ) / TD-Gammon
Métodos de temporal difference que combinan ideas de Monte Carlo y
programación dinámica para aprender de transiciones. Ver
[fase 9 — Aprendizaje por refuerzo](../fases/09-aprendizaje-por-refuerzo/README.md).

### Temperature (temperatura)
Parámetro de sampling que escala los logits antes del softmax.
- `T → 0`: greedy, determinista.
- `T → ∞`: uniforme, aleatorio.
- `T = 1`: distribución original. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Tensor
Array n-dimensional (generalización de vectores y matrices). Es la
estructura de datos fundamental en deep learning. Ver
[fase 1 — Fundamentos de matemáticas](../fases/01-fundamentos-matematicas/README.md).

### Token
Unidad mínima que procesa un LLM. Puede ser una palabra, sub-palabra o
carácter. Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

### Tokenizer
Componente que convierte texto crudo en una secuencia de tokens. Ver
[fase 10 — LLMs desde cero](../fases/10-llms-desde-cero/README.md).

### Tool use (uso de herramientas)
Capacidad de un LLM (vía function calling) de invocar
funciones/APIs externas y razonar con los resultados. Ver
[fase 13 — Herramientas y protocolos](../fases/13-herramientas-y-protocolos/README.md).

### TPU (Tensor Processing Unit)
ASIC desarrollado por Google específicamente para multiplicación de
matrices usada en redes neuronales. Alternativa a GPUs de NVIDIA. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

### Transformer
Arquitectura de red neuronal basada exclusivamente en mecanismos de
atención. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

### TTFT (Time To First Token)
Latencia desde que se envía el prompt hasta que el LLM emite el primer
token. Métrica clave de "responsiveness" en serving. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

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

### Transformer (arquitectura)
Ver **Transformer** arriba. Modelo basado exclusivamente en
mecanismos de atención. Ver
[fase 7 — Transformers a fondo](../fases/07-transformers-a-fondo/README.md).

---

## U

### Unsupervised learning (aprendizaje no supervisado)
Paradigma en el que el modelo aprende la estructura de los datos sin
etiquetas. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

---

## V

### VAE (Variational Autoencoder)
Autoencoder que aprende un prior Gaussiano sobre el espacio latente,
permitiendo muestreo y generación. Base de muchos modelos
generativos. Ver
[fase 8 — IA generativa](../fases/08-ia-generativa/README.md).

### Validation set (conjunto de validación)
Subconjunto de los datos separado del entrenamiento, usado para
ajustar hiperparámetros.

### Variance (varianza)
Sensibilidad del modelo a fluctuaciones en los datos de entrenamiento.
Varianza alta ⇒ sobreajuste.

### Vector database (base de datos vectorial)
Almacén optimizado para búsqueda por similitud de embeddings
(vectorial): Pinecone, Weaviate, Milvus, Qdrant, Chroma. Ver
[fase 11 — Ingeniería de LLMs](../fases/11-ingenieria-llms/README.md).

### ViT (Vision Transformer)
Aplicación del Transformer a imágenes mediante patches de tamaño
fijo (16x16) tratados como tokens. Ver
[fase 4 — Visión por computador](../fases/04-vision-por-computador/README.md).

### vLLM
Sistema de serving open-source de alto rendimiento para LLMs, basado
en PagedAttention y continuous batching. Ver
[fase 17 — Infraestructura y producción](../fases/17-infraestructura-y-produccion/README.md).

### Voice agent (agente de voz)
Sistema conversacional que combina ASR (entrada), LLM (razonamiento)
y TTS (salida) para interactuar por voz. Pipecat, LiveKit. Ver
[fase 14 — Ingeniería de agentes](../fases/14-ingenieria-agentes/README.md).

---

## W

### Wav2vec / Whisper
Modelos de ASR pre-entrenados con aprendizaje auto-supervisado
(Wav2vec) o weak-supervised (Whisper). Ver
[fase 6 — Voz y audio](../fases/06-voz-y-audio/README.md).

### Weight (peso)
Parámetro aprendible de un modelo. Una red neuronal tiene miles o
millones de pesos.

### Word2Vec / GloVe / fastText
Algoritmos clásicos para aprender embeddings de palabras a partir
de grandes corpus (Skip-gram, CBOW, coocurrencia global, subword). Ver
[fase 5 — NLP](../fases/05-nlp-fundamentos-a-avanzado/README.md).

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

## X

### XGBoost / LightGBM
Implementaciones eficientes del algoritmo de gradient boosting sobre
árboles de decisión. XGBoost es el más usado en ML clásico de
producción. Ver
[fase 2 — Fundamentos de ML](../fases/02-fundamentos-ml/README.md).

### XLA (Accelerated Linear Algebra)
Compilador de Google para optimizar cómputo de Tensores y JAX; usado
en TPUs. Ver
[fase 3 — Núcleo de Deep Learning](../fases/03-nucleo-deep-learning/README.md).

---

## Y

### YAML / TOML
Formatos de serialización legibles por humanos usados para
configuraciones y frontmatter de lecciones y skills.

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
