# Fase 14 — Ingeniería de agentes

> Construir agentes desde los primeros principios.

Esta es la **fase más densa** del currículo, y la que más cambió
entre 2024 y 2026. Un **agente** es un sistema que usa un LLM como
*cerebro* para decidir, planificar, ejecutar y revisar acciones
sobre el mundo. En 2026 los agentes dominan el panorama
productivo: Claude Code, Cursor 3, OpenCode, SWE-agent, Operator
de OpenAI, Computer Use de Anthropic, voice agents con Pipecat,
multi-agent crews con CrewAI. Esta fase recorre desde el *loop*
básico (plan → act → observe → recover) hasta workbenches de
producción con verificación y handoff multi-sesión.

La fase se organiza en **cinco bloques**. El **bloque 1** (lecciones
1–5) cubre los **patrones de razonamiento**: loop del agente,
ReWOO, Reflexion, Tree of Thoughts / LATS, Self-Refine. El
**bloque 2** (6–10) trata **memoria y herramientas**: tool use,
memoria virtual (MemGPT), memory blocks, memoria híbrida (Mem0) y
bibliotecas de skills (Voyager). El **bloque 3** (11–17) entra al
**planificación y frameworks**: HTN, patrones de Anthropic,
LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, Claude Agent SDK.
El **bloque 4** (18–30) cubre la **operación y evaluación**:
runtimes de producción, benchmarks, computer use, voice agents,
OpenTelemetry, observabilidad, multi-agente debate, modos de
falla, prompt injection, patrones de orquestación, runtimes de
cola/evento y desarrollo dirigido por evaluación. El **bloque 5**
(31–42) es el **workbench de agentes**: el patrón reusable para
construir agentes de coding confiables, con 12 lecciones que
describen cada componente del workbench.

## Índice de lecciones

### Bloque 1 — Patrones de razonamiento

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [El loop del agente](01-the-agent-loop/) | Construir | Plan-act-observe-recover, TodoWrite, hooks. |
| 02 | [ReWOO y plan-and-execute](02-rewoo-plan-and-execute/) | Construir | Planner + Worker + Solver desacoplados. |
| 03 | [Reflexion y verbal RL](03-reflexion-verbal-rl/) | Construir | Auto-crítica, memoria de reflexiones, *retry*. |
| 04 | [Tree of Thoughts y LATS](04-tree-of-thoughts-lats/) | Construir | Búsqueda en árbol con LLM como valor. |
| 05 | [Self-Refine y CRITIC](05-self-refine-and-critic/) | Construir | Iteración + crítica externa. |

### Bloque 2 — Memoria y herramientas

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [Tool use y function calling](06-tool-use-and-function-calling/) | Construir | Tools robustas, validación, *truncation*. |
| 07 | [Memoria: virtual context y MemGPT](07-memory-virtual-context-memgpt/) | Construir | Paginación de contexto, *function* tokens. |
| 08 | [Memory blocks y sleep-time compute](08-memory-blocks-sleep-time-compute/) | Construir | Bloques de memoria, *consolidation* periódico. |
| 09 | [Memoria híbrida: Mem0](09-hybrid-memory-mem0/) | Construir | Vector + graph + KV cache. |
| 10 | [Bibliotecas de skills: Voyager](10-skill-libraries-voyager/) | Construir | Skills reutilizables, *curriculum* automático. |

### Bloque 3 — Planificación y frameworks

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 11 | [Planning con HTN y búsqueda evolutiva](11-planning-htn-and-evolutionary/) | Construir | Jerarquías de tareas, *evolutionary search*. |
| 12 | [Patrones de workflow de Anthropic](12-anthropic-workflow-patterns/) | Construir | Prompt chaining, routing, parallelization. |
| 13 | [LangGraph — grafos con estado](13-langgraph-stateful-graphs/) | Construir | Nodos, edges, *checkpoints*, *human-in-the-loop*. |
| 14 | [AutoGen v0.4 — actor model](14-autogen-actor-model/) | Construir | Actores, *routed messages*, *subscriptions*. |
| 15 | [CrewAI — crews basadas en roles](15-crewai-role-based-crews/) | Construir | Roles, *tasks*, *crews*, *processes*. |
| 16 | [OpenAI Agents SDK](16-openai-agents-sdk/) | Construir | Handoffs, guardrails, tracing. |
| 17 | [Claude Agent SDK](17-claude-agent-sdk/) | Construir | Subagentes, session store. |

### Bloque 4 — Operación y evaluación

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 18 | [Agno y Mastra — runtimes de producción](18-agno-and-mastra-runtimes/) | Aprender | Runtimes tipados, *sessions*, *memory*. |
| 19 | [Benchmarks: SWE-bench, GAIA, AgentBench](19-benchmarks-swebench-gaia/) | Aprender | Métricas estándar. |
| 20 | [Benchmarks: WebArena y OSWorld](20-benchmarks-webarena-osworld/) | Aprender | Agentes en navegador y OS. |
| 21 | [Computer use: Claude, OpenAI CUA, Gemini](21-computer-use-agents/) | Construir | *Screenshot + action* loop. |
| 22 | [Voice agents: Pipecat y LiveKit](22-voice-agents-pipecat-livekit/) | Construir | STT → LLM → TTS en tiempo real. |
| 23 | [OpenTelemetry GenAI semantic conventions](23-otel-genai-conventions/) | Construir | *Spans* para tool calls y eval. |
| 24 | [Observabilidad de agentes: Langfuse, Phoenix, Opik](24-agent-observability-platforms/) | Aprender | *Tracing*, *eval*, *datasets*. |
| 25 | [Debate multi-agente y colaboración](25-multi-agent-debate/) | Construir | Múltiples agentes, *voting*, *judge*. |
| 26 | [Modos de falla: por qué se rompen los agentes](26-failure-modes-agentic/) | Construir | Catalogar y mitigar. |
| 27 | [Prompt injection y defensa PVE](27-prompt-injection-defense/) | Construir | *Prompt virtualization*, *canaries*. |
| 28 | [Patrones de orquestación: supervisor, swarm, jerárquico](28-orchestration-patterns/) | Construir | Topologías multi-agente. |
| 29 | [Runtimes de producción: queue, event, cron](29-production-runtimes/) | Aprender | Despliegue con *queues* y *events*. |
| 30 | [Desarrollo dirigido por evaluación](30-eval-driven-agent-development/) | Construir | *Evals* como red de seguridad. |

### Bloque 5 — Workbench de agentes

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 31 | [Agent workbench: por qué fallan los modelos capaces](31-agent-workbench-why-models-fail/) | Aprender | El problema central: *control* vs *capability*. |
| 32 | [El workbench mínimo de agentes](32-minimal-agent-workbench/) | Construir | Esqueleto del workbench. |
| 33 | [Instrucciones de agente como restricciones ejecutables](33-instructions-as-executable-constraints/) | Construir | Convertir prompts en *tests* y *linters*. |
| 34 | [Memoria de repo y estado durable](34-repo-memory-and-state/) | Construir | Persistir entre sesiones. |
| 35 | [Scripts de inicialización para agentes](35-initialization-scripts/) | Construir | *Bootstrap* del entorno. |
| 36 | [Contratos de scope y límites de tarea](36-scope-contracts/) | Construir | Definir qué hace y qué no hace el agente. |
| 37 | [Loops de feedback en runtime](37-runtime-feedback-loops/) | Construir | *Observe* y ajustar en tiempo real. |
| 38 | [Verification gates](38-verification-gates/) | Construir | Compuertas automáticas antes de *commit*. |
| 39 | [Agente revisor: separar builder de marker](39-reviewer-agent/) | Construir | Diff review con sub-agente. |
| 40 | [Handoff multi-sesión](40-multi-session-handoff/) | Construir | Transferir estado entre sesiones. |
| 41 | [El workbench sobre un repo real](41-workbench-for-real-repos/) | Construir | Caso aplicado. |
| 42 | [Capstone: ship un pack de workbench reutilizable](42-agent-workbench-capstone/) | Construir | Empaquetar y compartir. |

## Prerrequisitos

- **Fases 7, 9, 10, 11 y 13** completas (la Fase 13 idealmente
  en paralelo).
- Conocimiento sólido de function calling y de los frameworks
  comunes (LangChain, LlamaIndex).
- GPU con 24+ GB VRAM para fine-tuning de agentes pequeños
  (Phi-4, Qwen-Coder).
- Acceso a APIs de LLM o modelos locales con Ollama.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Implementar** el loop del agente (plan-act-observe-recover)
  desde cero y razonar sobre *hooks*, *truncation* y *sandboxing*.
- **Comparar** patrones de razonamiento (CoT, ReAct, ReWOO,
  Reflexion, ToT) y elegir el más adecuado para cada tarea.
- **Diseñar** sistemas de memoria (corto plazo, largo plazo,
  vector, grafo) y entender cuándo cada uno aporta valor.
- **Construir** un agente con LangGraph, AutoGen, CrewAI, OpenAI
  Agents SDK o Claude Agent SDK.
- **Desplegar** agentes en producción con observabilidad,
  guardrails, cost control y *eval-driven development*.
- **Mitigar** modos de falla conocidos: prompt injection,
  *context poisoning*, *reward hacking*, *runaway cost*.
- **Construir** un *agent workbench* reusable para un repo
  propio y publicarlo como pack.

## Stack y herramientas

- **OpenAI, Anthropic, Gemini, Mistral, Cohere, DeepSeek** como
  proveedores.
- **Ollama, vLLM, llama.cpp** para modelos locales.
- **LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, Claude Agent
  SDK, Agno, Mastra** como frameworks.
- **Pipecat, LiveKit, Daily** para voice agents.
- **Langfuse, Phoenix, Opik, Arize** para observabilidad.
- **OpenTelemetry GenAI** para instrumentación.
- **E2B, Daytona, Modal, Fly** para sandboxes.
- **SWE-bench, GAIA, AgentBench, WebArena, OSWorld** para
  evaluación.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Agent loop** | Lección 01 | Cualquier agente. |
| **ReWOO** | Lección 02 | Planificación desacoplada. |
| **Reflexion** | Lección 03 | Auto-mejora. |
| **MemGPT** | Lección 07 | Memoria virtual. |
| **LangGraph** | Lección 13 | Estado persistente. |
| **CrewAI** | Lección 15 | Roles y crews. |
| **Computer use** | Lección 21 | Agentes en pantalla. |
| **Voice agent** | Lección 22 | Agentes en tiempo real. |
| **OpenTelemetry** | Lección 23 | Trazabilidad. |
| **Verification gate** | Lección 38 | Control de calidad. |
| **Workbench** | Lecciones 31–42 | Patrón reusable. |

## Cómo estudiar esta fase

1. **La lección 01 (el loop) es la base de todo.** Sin entender
   plan-act-observe, las demás son variaciones.
2. **No te abrumes con todos los frameworks.** Elige UNO
   (LangGraph o CrewAI son los más versátiles) y domínalo.
3. **Las lecciones 26, 27 son obligatorias para producción.** Un
   agente sin defensa contra prompt injection es un riesgo de
   seguridad.
4. **El workbench (lecciones 31–42) es la parte más valiosa
   para ingeniería de coding agents.** Si tu tiempo es
   limitado, priorízalas.
5. **Evalúa desde el día uno.** La lección 30 (*eval-driven
   development*) debería ser hábito en cada agente que
   construyas.

## Verificación de progreso

```bash
# Lección 01 — el loop del agente
python3 fases/14-ingenieria-agentes/01-the-agent-loop/code/main.py

# Lección 13 — LangGraph con estado
python3 fases/14-ingenieria-agentes/13-langgraph-stateful-graphs/code/main.py

# Lección 42 — capstone del workbench
python3 fases/14-ingenieria-agentes/42-agent-workbench-capstone/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Topologías de multi-agente

| Patrón | Cuándo usarlo | Lección |
|---|---|---|
| Loop único | Tareas pequeñas | 01 |
| Plan-and-execute | Tareas largas, observaciones costosas | 02 |
| Reflexión iterativa | Tareas con auto-criterio | 03, 05 |
| Tree search | Tareas con búsqueda combinatorial | 04 |
| Supervisor | Coordinación central | 28 |
| Swarm | Agentes pares, mayoría | 25, 28 |
| Jerárquico | Equipos grandes, *scopes* claros | 28 |
| Debate | Validación, robustez | 25 |

## Conexión con otras fases

- **Entrada** → [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md),
  [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md) y
  [Fase 13 — Herramientas y protocolos](../13-herramientas-y-protocolos/README.md).
- **Salida natural** → [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md)
  y [Fase 16 — Multi-agente y enjambres](../16-multi-agente-y-enjambres/README.md).
- **Reuso en** → Fase 17 (producción), Fase 19 (capstone).

## Recursos recomendados

- *Designing Machine Learning Systems* — Huyen (cap. sobre agentes).
- *AI Agents in Action* — Micheal Lanham.
- *Anthropic — Building effective agents* — <https://anthropic.com>.
- *LangGraph docs* — <https://langchain-ai.github.io/langgraph>.
- *Claude Agent SDK* — <https://docs.anthropic.com>.
- *SWE-bench* — <https://www.swebench.com>.
- *pipecat docs* — <https://docs.pipecat.ai>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *agent*,
  *ReWOO*, *Reflexion*, *MemGPT*, *LangGraph*, *workbench*.
- [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- [Fase 13 — Herramientas y protocolos](../13-herramientas-y-protocolos/README.md).
- [Fase 16 — Multi-agente y enjambres](../16-multi-agente-y-enjambres/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
