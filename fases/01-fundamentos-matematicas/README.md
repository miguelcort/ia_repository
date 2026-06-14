# Fase 1 — Fundamentos de matemáticas

> La intuición detrás de cada algoritmo de IA, pasada por código.

Esta fase destila los **prerrequisitos matemáticos mínimos** que
necesitamos para seguir el currículo sin que la notación nos frene. No
es un curso de matemáticas: cada lección toma **una** idea, la conecta
con **una** operación de IA y la aterriza en **una** implementación
ejecutable en NumPy. La idea es que, cuando llegues a la Fase 3 y
derives backpropagation a mano, ya tengas el vocabulario y la
intuición geométrica suficientes para no atascarte en la notación.

El recorrido sigue la lógica de uso real: empezamos con **álgebra
lineal** (el lenguaje de los tensores), seguimos con **cálculo y
optimización** (la herramienta para entrenar), pasamos por
**probabilidad y estadística** (la base de la inferencia), y
cerramos con temas especializados que reaparecerán en fases
posteriores: teoría de la información, SVD, números complejos,
Fourier, grafos y procesos estocásticos. Cada bloque está pensado
para ser *suficiente* — no exhaustivo — y dejar al estudiante con la
confianza de leer un paper y entender la matemática que contiene.

## Índice de lecciones

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Intuición de álgebra lineal](01-intuicion-algebra-lineal/) | Aprender | Vectores y matrices como flechas y transformaciones geométricas. |
| 02 | [Vectores, matrices y operaciones](02-vectores-matrices-operaciones/) | Construir | Suma, producto punto, producto matricial, broadcasting. |
| 03 | [Transformaciones y valores propios](03-transformaciones-valores-propios/) | Construir | Autovalores/autovectores y diagonalización. |
| 04 | [Cálculo para ML: derivadas y gradientes](04-calculo-para-ml/) | Construir | Derivada, derivada parcial, gradiente, derivada direccional. |
| 05 | [Regla de la cadena y diferenciación automática](05-regla-de-la-cadena-y-autodiff/) | Construir | Backprop implementado a mano con grafos computacionales. |
| 06 | [Probabilidad y distribuciones](06-probabilidad-y-distribuciones/) | Aprender | Variables aleatorias, esperanza, varianza, distribuciones clave. |
| 07 | [Teorema de Bayes y pensamiento estadístico](07-bayes-y-pensamiento-estadistico/) | Construir | Actualización bayesiana y diagnóstico de modelos. |
| 08 | [Optimización: familia del descenso por gradiente](08-optimizacion-familia-gradiente/) | Construir | GD batch, mini-batch, estocástico, momentum, Adam. |
| 09 | [Teoría de la información: entropía, KL](09-teoria-de-la-informacion/) | Aprender | Entropía, divergencia KL, entropía cruzada. |
| 10 | [Reducción de dimensionalidad: PCA, t-SNE, UMAP](10-reduccion-de-dimensionalidad/) | Construir | PCA desde la covarianza, t-SNE/UMAP con scikit-learn. |
| 11 | [Descomposición en valores singulares (SVD)](11-descomposicion-svd/) | Construir | SVD como herramienta de compresión y recomendación. |
| 12 | [Operaciones con tensores](12-operaciones-con-tensores/) | Construir | `reshape`, `permute`, `broadcast` y `einsum` con NumPy. |
| 13 | [Estabilidad numérica](13-estabilidad-numerica/) | Construir | Overflow, underflow, `log-sum-exp`, comparación con épsilon. |
| 14 | [Normas y distancias](14-normas-y-distancias/) | Construir | L1, L2, coseno, Mahalanobis. |
| 15 | [Estadística para ML](15-estadistica-para-ml/) | Construir | Estimación, intervalos de confianza, bootstrap. |
| 16 | [Métodos de muestreo](16-metodos-de-muestreo/) | Construir | Monte Carlo, importance sampling, rechazo. |
| 17 | [Sistemas lineales](17-sistemas-lineales/) | Construir | Eliminación gaussiana, LU, condicionamiento. |
| 18 | [Optimización convexa](18-optimizacion-convexa/) | Construir | Convexidad, multiplicadores de Lagrange, KKT. |
| 19 | [Números complejos para IA](19-numeros-complejos/) | Aprender | Aritmética compleja y aplicación a señales. |
| 20 | [La transformada de Fourier](20-transformada-fourier/) | Construir | FFT 1D/2D, convolución como multiplicación espectral. |
| 21 | [Teoría de grafos para ML](21-teoria-de-grafos/) | Construir | PageRank, random walks, GNN-ready. |
| 22 | [Procesos estocásticos](22-procesos-estocasticos/) | Aprender | Cadenas de Markov, caminata aleatoria, martingalas. |

## Prerrequisitos

- **Fase 0 completa** (entorno, Git, Python, Docker).
- Conocimiento básico de álgebra de bachillerato (operaciones con
  polinomios, trigonometría elemental).
- Curiosidad y disposición a *verificar* cada identidad con código.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Leer** la notación de un paper de ML (sumatorios, derivadas
  parciales, normas) y traducirla a una operación de NumPy.
- **Derivar** a mano la regla de actualización de un modelo de
  regresión logística y verificarla con un *autograd* minimal.
- **Implementar** descenso por gradiente, PCA y SVD desde cero y
  compararlos con las versiones de scikit-learn.
- **Diagnosticar** problemas numéricos comunes: explosiones de
  gradiente, matrices singulares, *NaN* por overflow.
- **Justificar** con intuición geométrica por qué la atención de
  los transformers es un producto punto y por qué el *softmax*
  aparece una y otra vez.

## Stack y herramientas

- **NumPy** como única dependencia numérica (sin PyTorch todavía).
- **Matplotlib** para visualizar distribuciones y trayectorias de
  optimización.
- **Jupyter** opcional para experimentar; los tests son la
  verificación canónica.
- **scikit-learn** sólo como referencia en la lección 10 (PCA,
  t-SNE, UMAP).

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Producto punto** | Lecciones 1, 2, 3, 14 | Fase 3 (backprop), Fase 7 (atención) |
| **Gradiente** | Lecciones 4, 5, 8 | Fase 3 (entrenamiento), Fase 9 (policy gradient) |
| **Distribución gaussiana** | Lecciones 6, 15, 16 | Fase 4 (detección de anomalías) |
| **Entropía cruzada** | Lecciones 9, 6 | Fase 3 (clasificación), Fase 10 (LLM loss) |
| **SVD** | Lección 11 | Fase 5 (LSA), Fase 12 (LoRA) |
| **FFT** | Lección 20 | Fase 6 (audio), Fase 4 (convolución rápida) |
| **Cadenas de Markov** | Lección 22 | Fase 9 (MDPs), Fase 8 (diffusion) |

## Cómo estudiar esta fase

1. **No memorices fórmulas.** Cada lección entrega una implementación
   en `code/main.py` con sus tests. La memoria operativa del
   estudiante debe ser "leer la fórmula → correr el test → comparar
   con la librería".
2. **Resuelve los ejercicios a mano antes de programarlos.** Las
   primeras cinco lecciones asumen que el estudiante está cómodo
   con operaciones matriciales *en papel*. Programa después.
3. **Reusa el cuaderno de una lección** como sandbox para la
   siguiente. La lección 08 (gradiente) reutiliza el grafo
   computacional de la 05.
4. **Lleva un glosario personal.** Aunque
   [glosario/terminos.md](../../glosario/terminos.md) tiene los
   términos canónicos, escribir la definición con tus propias
   palabras fija el concepto.

## Verificación de progreso

```bash
# Lección 05 — verificar backprop a mano
python3 fases/01-fundamentos-matematicas/05-regla-de-la-cadena-y-autodiff/code/main.py

# Lección 08 — comparar GD contra scipy.optimize
python3 fases/01-fundamentos-matematicas/08-optimizacion-familia-gradiente/code/main.py

# Lección 11 — SVD y reconstrucción de bajo rango
python3 fases/01-fundamentos-matematicas/11-descomposicion-svd/code/main.py
```

Si los tres demos terminan con código 0 y las salidas coinciden con
las de la lección, el estudiante está listo para la Fase 2.

## Conexión con otras fases

- **Entrada** → [Fase 0 — Configuración y herramientas](../00-configuracion-y-herramientas/README.md).
- **Salida natural** → [Fase 2 — Fundamentos de ML](../02-fundamentos-ml/README.md)
  (los algoritmos de la fase 2 aplican directamente la teoría de esta fase).
- **Reuso en fases futuras** → atención (Fase 7), RL (Fase 9), LoRA
  (Fase 11), SDEs (Fase 8), convolución (Fase 4).

## Recursos recomendados

- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong (PDF libre).
- *3Blue1Brown — Essence of Linear Algebra* (YouTube).
- *3Blue1Brown — Essence of Calculus* (YouTube).
- *Khan Academy — Statistics & Probability*.
- *Pattern Recognition and Machine Learning* — Bishop, cap. 1–2.
- *Convex Optimization* — Boyd & Vandenberghe (PDF libre).

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — definiciones canónicas.
- [PLANTILLA_LECCION.md](../../PLANTILLA_LECCION.md) — cómo se estructura cada lección.
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
