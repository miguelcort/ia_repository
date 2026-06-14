# Fase 11 — Ingeniería de LLMs en producción

> Llevar los LLMs al mundo real.

Un **LLM en producción** es mucho más que una llamada a
`openai.ChatCompletion.create()`. Es un sistema con prompts
cuidadosamente diseñados, RAG para conocimiento actualizado,
*fine-tuning* con LoRA para especializar el modelo, function calling
para invocar herramientas, guardrails para evitar salidas dañinas,
caching para controlar costo, observabilidad para diagnosticar
errores, y un pipeline de evaluación continua. Esta fase cubre
todos esos componentes y entrega un patrón reusable para
**aplicaciones LLM de grado de producción**.

La fase se organiza en **cuatro bloques**. El **bloque 1** (lecciones
1–5) cubre la **interfaz con el LLM**: prompt engineering, few-shot
y chain-of-thought, salidas estructuradas, embeddings y context
engineering. El **bloque 2** (6–9) entra al **razonamiento sobre
datos externos**: RAG básico, RAG avanzado (chunking, re-ranking),
fine-tuning con LoRA/QLoRA y function calling. El **bloque 3**
(10–13) trata la **operación**: evaluación, caching, guardrails y
una app completa de producción. El **bloque 4** (14–17) cierra con
**protocolos y frameworks**: MCP, prompt caching, LangGraph y
comparativa de frameworks de agentes.

## Índice de lecciones

### Bloque 1 — Interfaz con el LLM

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Prompt engineering: técnicas y patrones](01-prompt-engineering/) | Construir | Roles, delimitadores, instrucciones negativas, *meta-prompting*. |
| 02 | [Few-shot, Chain-of-Thought, Tree-of-Thought](02-few-shot-y-cot/) | Construir | Razonamiento explícito, *self-consistency*. |
| 03 | [Salidas estructuradas](03-outputs-estructurados/) | Construir | JSON mode, *grammar-constrained decoding*, *logit processors*. |
| 04 | [Embeddings y representaciones vectoriales](04-embeddings/) | Construir | Sentence-BERT, E5, BGE, *matryoshka*. |
| 05 | [Context engineering](05-context-engineering/) | Construir | Composición de contexto, *token budgeting*. |

### Bloque 2 — Razonamiento sobre datos externos

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [RAG: Retrieval-Augmented Generation](06-rag/) | Construir | Pipeline básico con embeddings y FAISS. |
| 07 | [RAG avanzado](07-rag-avanzado/) | Construir | Chunking, *hybrid search*, *re-ranking*, *query rewriting*. |
| 08 | [Fine-tuning con LoRA y QLoRA](08-fine-tuning-lora/) | Construir | Adaptación eficiente con `peft` y `bitsandbytes`. |
| 09 | [Function calling y tool use](09-function-calling/) | Construir | JSON schemas, *parallel tool use*, validación. |

### Bloque 3 — Operación y producción

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 10 | [Evaluación y testing](10-evaluacion/) | Construir | RAGAS, DeepEval, *llm-as-judge*, regression tests. |
| 11 | [Caching, rate limiting y costo](11-caching-y-costo/) | Construir | Prompt cache, *semantic cache*, *token budgeting*. |
| 12 | [Guardrails y seguridad](12-guardrails/) | Construir | Validación de input/output, *jailbreak defense*, PII. |
| 13 | [Construir una app LLM en producción](13-aplicacion-de-produccion/) | Construir | FastAPI + RAG + observabilidad + guardrails. |

### Bloque 4 — Protocolos y frameworks

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 14 | [Model Context Protocol (MCP)](14-model-context-protocol/) | Construir | Servidor y cliente MCP, *tools*, *resources*, *prompts*. |
| 15 | [Prompt caching y context caching](15-prompt-caching/) | Construir | Anthropic, OpenAI, Gemini caches. |
| 16 | [LangGraph: state machines para agentes](16-langgraph-y-state-machines/) | Construir | Grafos con estado, *checkpoints*, *human-in-the-loop*. |
| 17 | [Tradeoffs de frameworks de agentes](17-tradeoffs-de-frameworks-de-agentes/) | Aprender | LangGraph, AutoGen, CrewAI, OpenAI Agents SDK. |

## Prerrequisitos

- **Fase 10** completa.
- Conocimiento de Python, FastAPI o similar.
- Familiaridad con Docker (Fase 0).
- Acceso a una API de LLM (OpenAI, Anthropic, Gemini, etc.) o
  modelos locales con Ollama.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Diseñar** prompts efectivos con técnicas de few-shot, CoT y
  *self-consistency*.
- **Construir** un sistema RAG completo con embeddings, retrieval,
  re-ranking y evaluación.
- **Hacer fine-tuning** de un modelo open-weight con LoRA/QLoRA.
- **Implementar** function calling con JSON schemas y validación
  de argumentos.
- **Desplegar** una app LLM con FastAPI, observabilidad,
  guardrails y caching.
- **Construir** un servidor MCP para exponer herramientas a un
  agente.
- **Evaluar** aplicaciones LLM con RAGAS, DeepEval y *llm-as-judge*.
- **Medir y optimizar** latencia, costo y calidad.

## Stack y herramientas

- **OpenAI, Anthropic, Gemini, Mistral, Cohere** como proveedores.
- **Ollama, vLLM, llama.cpp** para modelos locales.
- **LangChain, LlamaIndex** como frameworks (con criterio).
- **LangGraph** para stateful agents.
- **PEFT, bitsandbytes, axolotl** para fine-tuning.
- **RAGAS, DeepEval, Phoenix, Langfuse** para evaluación y
  observabilidad.
- **FastAPI, Uvicorn, Pydantic** para servir.
- **Pinecone, Weaviate, Qdrant, Chroma** para vector stores.
- **Guardrails AI, NeMo Guardrails, Llama Guard** para seguridad.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Prompt engineering** | Lección 01 | Toda la fase. |
| **CoT / ToT** | Lección 02 | Razonamiento. |
| **Embeddings** | Lección 04 | RAG, búsqueda, clustering. |
| **RAG** | Lecciones 06, 07 | Agentes, chatbots. |
| **LoRA / QLoRA** | Lección 08 | Personalización. |
| **Function calling** | Lección 09 | Agentes, MCP. |
| **Guardrails** | Lección 12 | Producción. |
| **MCP** | Lección 14 | Agentes. |
| **LangGraph** | Lección 16 | Agentes stateful. |

## Cómo estudiar esta fase

1. **Comienza por el prompt engineering.** Es la técnica más
   rentable: 80% de los problemas de LLM se resuelven con mejor
   prompting.
2. **RAG antes que fine-tuning.** Si tu problema es "el modelo
   no sabe sobre X", RAG es la primera opción. Fine-tuning
   viene después.
3. **Mide desde el día uno.** La evaluación (lección 10) no es
   opcional: sin métricas, no sabes si tu cambio mejora o
   empeora.
4. **Function calling es la base de los agentes.** La lección
   09 es prerequisito conceptual de la Fase 14.
5. **No reinventes la rueda.** MCP (lección 14) es un estándar;
   úsalo en lugar de inventar tu propio protocolo de tools.

## Verificación de progreso

```bash
# Lección 06 — RAG básico
python3 fases/11-ingenieria-llms/06-rag/code/main.py

# Lección 09 — function calling
python3 fases/11-ingenieria-llms/09-function-calling/code/main.py

# Lección 13 — app de producción
python3 fases/11-ingenieria-llms/13-aplicacion-de-produccion/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Patrón de referencia: una app LLM en producción

```text
[Usuario] → [API Gateway]
              ↓
        [Cache check]
              ↓ miss
        [Input guardrail]
              ↓
        [Context retrieval (RAG)]
              ↓
        [LLM (with structured output)]
              ↓
        [Output guardrail]
              ↓
        [Cache write]
              ↓
        [Observability log]
              ↓
        [Respuesta]
```

Este es el esqueleto que la lección 13 implementa paso a paso.

## Conexión con otras fases

- **Entrada** → [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- **Salida natural** → [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- **Reuso en** → Fase 12 (multimodal), Fase 17 (infraestructura),
  Fase 19 (capstone).

## Recursos recomendados

- *Designing Machine Learning Systems* — Huyen (cap. sobre LLMs).
- *Prompt Engineering for LLMs* — Berryman & Zubiaga.
- *LangChain / LlamaIndex docs* — <https://docs.langchain.com>.
- *Hugging Face SFT docs* — <https://huggingface.co/docs/trl>.
- *Anthropic prompt engineering guide* — <https://docs.anthropic.com>.
- *OpenAI function calling guide* — <https://platform.openai.com>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *prompt*,
  *RAG*, *fine-tuning*, *MCP*, *guardrails*.
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [Fase 17 — Infraestructura y producción](../17-infraestructura-y-produccion/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
