# Fase 19 — Proyectos capstone

> Proyecto integrador que demuestra competencia de extremo a extremo.

Esta fase es la **entrega final** del currículo. El estudiante
demuestra competencia de extremo a extremo eligiendo, diseñando e
implementando uno o más **proyectos capstone** que integran
conceptos de ciencia de datos, modelado, despliegue, ética y
comunicación. La fase no es un solo proyecto: es un **menú
amplio** de 87 lecciones-capstone organizadas en 6 bloques, que
van desde agentes de coding hasta pipelines de RLHF, sistemas
multimodales, multi-agente y auditorías de seguridad. Cada
estudiante elige 1–3 proyectos según su tiempo y ambición.

La fase se organiza en **seis bloques**. El **bloque 1**
(lecciones 1–17) cubre **agentes de coding y de producto**:
terminal-native coding agent, RAG over codebase, voice
assistant, multimodal document QA, autonomous research agent,
devops troubleshooting, end-to-end fine-tuning, production RAG
chatbot, code migration, multi-agent software team, LLM
observability, video understanding, MCP server con registry,
speculative decoding server, constitutional safety harness,
GitHub issue-to-PR y personal AI tutor. El **bloque 2** (20–29)
es el **agent harness y workbench** (10 lecciones). El **bloque
3** (30–40) cubre el **GPT desde cero**: tokenizer BPE, sliding
window, multi-head attention, transformer block, training loop,
pretrained weights, classifier finetuning, instruction tuning,
DPO y eval pipeline. El **bloque 4** (41–49) trata **datos y
entrenamiento distribuido**: large corpus downloader, HDF5
corpus, cosine LR + warmup, gradient clipping + AMP, gradient
accumulation, checkpoint save/resume, FSDP/DDP, LM eval
harness. El **bloque 5** (50–63) cubre **investigación
autónoma y multimodal** (14 lecciones). El **bloque 6** (64–87)
cierra con **RAG avanzado, eval, distribuido y safety
(24 lecciones)**.

## Índice de lecciones

### Bloque 1 — Agentes de coding y de producto (17 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Terminal-native coding agent](01-terminal-native-coding-agent/) | Construir | TUI harness, plan-act-observe, sandbox, hooks. |
| 02 | [RAG over codebase](02-rag-over-codebase/) | Construir | Búsqueda semántica sobre el código del usuario. |
| 03 | [Realtime voice assistant](03-realtime-voice-assistant/) | Construir | STT → LLM → TTS en tiempo real. |
| 04 | [Multimodal document QA](04-multimodal-document-qa/) | Construir | PDF + imagen + tabla, retrieval y respuesta. |
| 05 | [Autonomous research agent](05-autonomous-research-agent/) | Construir | Búsqueda, lectura, síntesis, citación. |
| 06 | [DevOps troubleshooting agent](06-devops-troubleshooting-agent/) | Construir | Logs, métricas, kubectl, runbooks. |
| 07 | [End-to-end fine-tuning pipeline](07-end-to-end-fine-tuning-pipeline/) | Construir | Datos → SFT → DPO → eval → deploy. |
| 08 | [Production RAG chatbot](08-production-rag-chatbot/) | Construir | RAG con caching, observabilidad, eval. |
| 09 | [Code migration agent](09-code-migration-agent/) | Construir | Migrar de Python 2 a 3, jQuery a React, etc. |
| 10 | [Multi-agent software team](10-multi-agent-software-team/) | Construir | PM, dev, QA, reviewer en *crew*. |
| 11 | [LLM observability dashboard](11-llm-observability-dashboard/) | Construir | Métricas, *traces*, costos, *evals*. |
| 12 | [Video understanding pipeline](12-video-understanding-pipeline/) | Construir | VLM + transcripción + resumen. |
| 13 | [MCP server with registry](13-mcp-server-with-registry/) | Construir | Servidor MCP + catálogo de tools. |
| 14 | [Speculative decoding server](14-speculative-decoding-server/) | Construir | Draft + verify a alta concurrencia. |
| 15 | [Constitutional safety harness](15-constitutional-safety-harness/) | Construir | Critique-revise y guardrails. |
| 16 | [GitHub issue-to-PR agent](16-github-issue-to-pr-agent/) | Construir | Issue → branch → PR → review. |
| 17 | [Personal AI tutor](17-personal-ai-tutor/) | Construir | Asistente adaptativo al estudiante. |

### Bloque 2 — Agent harness y workbench (10 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 20 | [Agent harness loop contract](20-agent-harness-loop-contract/) | Construir | Contrato del *loop* del agente. |
| 21 | [Tool registry schema validation](21-tool-registry-schema-validation/) | Construir | Registro de *tools* con JSON Schema. |
| 22 | [JSON-RPC stdio transport](22-jsonrpc-stdio-transport/) | Construir | Transporte base para MCP. |
| 23 | [Function call dispatcher](23-function-call-dispatcher/) | Construir | Despachador tipado de function calls. |
| 24 | [Plan-execute control flow](24-plan-execute-control-flow/) | Construir | Patrón ReWOO en producción. |
| 25 | [Verification gates & observation budget](25-verification-gates-observation-budget/) | Construir | Compuertas y *budgets*. |
| 26 | [Sandbox runner deny-list](26-sandbox-runner-denylist/) | Construir | Sandbox con *deny list* y permisos. |
| 27 | [Eval harness fixture tasks](27-eval-harness-fixture-tasks/) | Construir | Tareas *fixture* reproducibles. |
| 28 | [Observability OTel traces](28-observability-otel-traces/) | Construir | Trazas OpenTelemetry. |
| 29 | [End-to-end coding task demo](29-end-to-end-coding-task-demo/) | Construir | Demo integral del workbench. |

### Bloque 3 — GPT desde cero (10 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 30 | [BPE tokenizer from scratch](30-bpe-tokenizer-from-scratch/) | Construir | Tokenizer BPE en 200 líneas. |
| 31 | [Tokenized dataset sliding window](31-tokenized-dataset-sliding-window/) | Construir | *Dataset* tokenizado y *sliding window*. |
| 32 | [Token + positional embeddings](32-token-positional-embeddings/) | Construir | *Embeddings* y *positional encodings*. |
| 33 | [Multi-head self-attention](33-multihead-self-attention/) | Construir | Atención multi-cabezal desde cero. |
| 34 | [Transformer block](34-transformer-block/) | Construir | Bloque transformer completo. |
| 35 | [GPT model assembly](35-gpt-model-assembly/) | Construir | Apilar bloques en un GPT. |
| 36 | [Training loop & eval](36-training-loop-eval/) | Construir | Loop de entrenamiento y evaluación. |
| 37 | [Loading pretrained weights](37-loading-pretrained-weights/) | Construir | Cargar pesos de GPT-2 o similar. |
| 38 | [Classifier finetuning](38-classifier-finetuning/) | Construir | Fine-tuning para clasificación. |
| 39 | [Instruction tuning (SFT)](39-instruction-tuning-sft/) | Construir | SFT con datos Alpaca. |
| 40 | [DPO from scratch](40-dpo-from-scratch/) | Construir | DPO en PyTorch. |
| 41 | [Eval pipeline](41-eval-pipeline/) | Construir | Pipeline de evaluación reproducible. |

### Bloque 4 — Datos y entrenamiento distribuido (9 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 42 | [Large corpus downloader](42-large-corpus-downloader/) | Construir | Descargador con *resumable downloads*. |
| 43 | [HDF5 tokenized corpus](43-hdf5-tokenized-corpus/) | Construir | Almacenamiento eficiente con HDF5. |
| 44 | [Cosine LR + warmup](44-cosine-lr-warmup/) | Construir | Scheduler con *cosine annealing*. |
| 45 | [Gradient clipping + AMP](45-gradient-clipping-amp/) | Construir | Precisión mixta y *clipping*. |
| 46 | [Gradient accumulation](46-gradient-accumulation/) | Construir | *Effective batch* con memoria limitada. |
| 47 | [Checkpoint save/resume](47-checkpoint-save-resume/) | Construir | Resiliencia ante fallos. |
| 48 | [Distributed FSDP/DDP](48-distributed-fsdp-ddp/) | Construir | Entrenamiento multi-GPU/multi-nodo. |
| 49 | [LM eval harness](49-lm-eval-harness/) | Construir | Evaluación con EleutherAI harness. |

### Bloque 5 — Investigación autónoma y multimodal (14 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 50 | [Hypothesis generator](50-hypothesis-generator/) | Construir | Generador de hipótesis científicas. |
| 51 | [Literature retrieval](51-literature-retrieval/) | Construir | Búsqueda en arXiv, Semantic Scholar. |
| 52 | [Experiment runner](52-experiment-runner/) | Construir | Runner con *sandbox* y *seeds*. |
| 53 | [Result evaluator](53-result-evaluator/) | Construir | Métricas y análisis estadístico. |
| 54 | [Paper writer](54-paper-writer/) | Construir | Borrador de paper con citas. |
| 55 | [Critic loop](55-critic-loop/) | Construir | Crítica iterativa del paper. |
| 56 | [Iteration scheduler](56-iteration-scheduler/) | Construir | Scheduler de iteraciones. |
| 57 | [End-to-end research demo](57-end-to-end-research-demo/) | Construir | Demo AI-Scientist. |
| 58 | [Vision encoder patches](58-vision-encoder-patches/) | Construir | ViT para multimodal. |
| 59 | [ViT transformer](59-vit-transformer/) | Construir | ViT completo. |
| 60 | [Projection layer modality align](60-projection-layer-modality-align/) | Construir | Proyector vision→LLM. |
| 61 | [Cross-attention fusion](61-cross-attention-fusion/) | Construir | Fusión cross-modal. |
| 62 | [Vision-language pretraining](62-vision-language-pretraining/) | Construir | Pre-training VLM. |
| 63 | [Multimodal eval](63-multimodal-eval/) | Construir | Métricas de VLM. |

### Bloque 6 — RAG avanzado, eval, distribuido y safety (24 lecciones)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 64 | [Chunking strategies advanced](64-chunking-strategies-advanced/) | Construir | Semantic, hierarchical, late chunking. |
| 65 | [Hybrid retrieval BM25 + dense](65-hybrid-retrieval-bm25-dense/) | Construir | Recuperación híbrida. |
| 66 | [Reranker cross-encoder](66-reranker-cross-encoder/) | Construir | Re-ranking con cross-encoder. |
| 67 | [Query rewriting / HyDE](67-query-rewriting-hyde/) | Construir | Re-escritura de queries. |
| 68 | [RAG eval precision/recall](68-rag-eval-precision-recall/) | Construir | Evaluación de RAG. |
| 69 | [End-to-end RAG system](69-end-to-end-rag-system/) | Construir | RAG production-ready. |
| 70 | [Task spec format](70-task-spec-format/) | Construir | Formato de spec para evals. |
| 71 | [Classical metrics](71-classical-metrics/) | Construir | Accuracy, F1, EM, F2. |
| 72 | [Code exec metric](72-code-exec-metric/) | Construir | Ejecución de código como métrica. |
| 73 | [Perplexity & calibration](73-perplexity-calibration/) | Construir | PPL y ECE. |
| 74 | [Leaderboard aggregation](74-leaderboard-aggregation/) | Construir | Agregación con Bootstrap. |
| 75 | [End-to-end eval runner](75-end-to-end-eval-runner/) | Construir | Runner con *caching* y *resume*. |
| 76 | [Collective ops from scratch](76-collective-ops-from-scratch/) | Construir | All-reduce, all-gather. |
| 77 | [Data parallel DDP](77-data-parallel-ddp/) | Construir | DDP en PyTorch. |
| 78 | [Zero parameter sharding](78-zero-parameter-sharding/) | Construir | ZeRO-1/2/3. |
| 79 | [Pipeline parallel](79-pipeline-parallel/) | Construir | GPipe, PipeDream. |
| 80 | [Checkpoint sharded resume](80-checkpoint-sharded-resume/) | Construir | *Resharding* al *resume*. |
| 81 | [End-to-end distributed train](81-end-to-end-distributed-train/) | Construir | Pipeline distribuido completo. |
| 82 | [Jailbreak taxonomy](82-jailbreak-taxonomy/) | Construir | Taxonomía de jailbreaks. |
| 83 | [Prompt injection detector](83-prompt-injection-detector/) | Construir | Detector entrenado. |
| 84 | [Refusal evaluation](84-refusal-evaluation/) | Construir | Evaluar *refusals* correctos. |
| 85 | [Content classifier integration](85-content-classifier-integration/) | Construir | Integración con clasificador. |
| 86 | [Constitutional rules engine](86-constitutional-rules-engine/) | Construir | Motor de reglas constitucionales. |
| 87 | [End-to-end safety gate](87-end-to-end-safety-gate/) | Construir | Pipeline completo de safety. |

## Prerrequisitos

- **Fases 0–18** completas (recomendado: todas).
- Mínimo: fases 0, 2, 3, 4, 9, 17 y 18.
- Conocimiento de PyTorch, FastAPI, Docker, Kubernetes.
- GPU con 24+ GB VRAM para los proyectos de entrenamiento.
- Capacidad de invertir 35+ horas por proyecto.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Diseñar** un proyecto de IA de extremo a extremo: problema,
  datos, modelado, producción, ética, comunicación.
- **Implementar** al menos un capstone completo de los 87
  disponibles, o una combinación de 2–3 más pequeños.
- **Medir** el éxito del proyecto con métricas claras y una
  rúbrica explícita.
- **Comunicar** los resultados con un README, notebook narrativo
  y presentación.
- **Auditar** el proyecto éticamente: sesgos, explicabilidad,
  privacidad, alineación.
- **Iterar** sobre el trabajo con feedback de pares y *evals*
  automatizados.

## Estructura sugerida del proyecto

1. **Problema y datos.** Definición, adquisición, EDA,
   consideraciones éticas desde el día uno.
2. **Modelado.** Baseline, modelo final, métricas, validación,
   ablación.
3. **Producción.** Docker, endpoint, monitoreo, observabilidad,
   FinOps.
4. **Ética.** Análisis de sesgos, explicabilidad (SHAP/LIME),
   privacidad (DP), *red team*.
5. **Comunicación.** README, notebook narrativo, *demo* en
   vivo, presentación.

## Rúbrica de evaluación

| Peso | Criterio | Cómo se mide |
|:-:|---|---|
| 25 | Problema y motivación | Claridad, relevancia, novelty. |
| 20 | Solución técnica | Calidad del modelo, baselines, ablación. |
| 20 | Producción | Despliegue, observabilidad, reproducibilidad. |
| 15 | Ética | Sesgos, privacidad, *red team*, *guardrails*. |
| 10 | Comunicación | README, *demo*, presentación. |
| 10 | Iteración y *evals* | Uso de *evals* automatizados y feedback. |
| **100** | | |

## Ejemplos de proyectos sobresalientes

- **Coding agent comparativo:** implementa 2–3 harnesses
  diferentes (Claude Code, OpenCode, custom) y evalúalos en
  SWE-bench Pro.
- **Voice agent en producción:** STT + LLM + TTS con
  observabilidad, guardrails y métricas de UX.
- **Multimodal document QA:** sistema RAG multimodal con
  ColPali y OCR para una industria específica.
- **DevOps troubleshooting agent:** agente que diagnostica
  incidentes usando logs, métricas y runbooks.
- **GPT desde cero:** entrena un mini-GPT en TinyStories y
  publícalo con un playground.
- **Distributed training pipeline:** pre-entrena un modelo
  pequeño con FSDP en un cluster multi-GPU.
- **AI Scientist replicado:** reproduce AI Scientist v2 sobre
  un dominio de investigación específico.
- **Safety pipeline completo:** integra Constitutional AI,
  Llama Guard, watermark y eval de jailbreaks.

## Conexión con otras fases

- **Entrada** → Todas las fases previas.
- **Salida natural** → No hay salida: esta es la fase final.
- **Reuso en** → Portfolio personal, contribuciones open
  source, productos comerciales.

## Recursos recomendados

- *The Capstone Project Handbook* — genérico.
- *Anthropic Cookbook* — patrones de agentes en producción.
- *SWE-bench Pro* — para evaluar coding agents.
- *AI Scientist v2 paper* — Sakana AI, 2025.
- *Aider, OpenCode, Claude Code* — referencias de coding
  agents.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *capstone*,
  *RAG*, *coding agent*, *safety pipeline*.
- [Todas las fases previas](../README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
