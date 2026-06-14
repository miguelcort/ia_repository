# Fase 16 — Multi-agente y enjambres

> Coordinación, protocolos, negociación y simulación de muchos agentes.

La fase anterior construyó **agentes individuales**. Esta fase los
**combina**: multi-agente con supervisor, enjambre paralelo,
jerarquías, *blackboard* compartido, consenso Bizantino, votación,
negociación, simulaciones tipo *Generative Agents*, teoría de la
mente, *swarm optimization* (PSO/ACO), MARL clásico (MADDPG, QMIX,
MAPPO), economías de agentes, escalado en producción y análisis de
modos de falla. La fase también introduce **protocolos formales**:
FIPA-ACL, A2A (Agent-to-Agent), patrones de handoff, y los
*frameworks* más usados en 2026 (CrewAI, AutoGen, LangGraph).

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–4) sienta las **bases conceptuales**: por qué
multi-agente, herencia FIPA-ACL, protocolos de comunicación y
*primitive model*. El **bloque 2** (5–11) cubre los **patrones
arquitectónicos**: supervisor, jerárquico, *society of mind*,
roles, swarm paralelo, group chat y handoffs. El **bloque 3**
(12–18) trata **protocolos formales y coordinación**: A2A,
*blackboard*, consenso Bizantino, votación, negociación,
*generative agents* y *theory of mind*. El **bloque 4** (19–25)
cierra con **swarm optimization, MARL y producción**: PSO/ACO,
MADDPG/QMIX/MAPPO, economías de agentes, escalado, modos de
falla y casos de estudio 2026.

## Índice de lecciones

### Bloque 1 — Bases conceptuales

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Why multi-agent](01-why-multi-agent/) | Construir | Cuándo dos agentes son mejor que uno. |
| 02 | [FIPA ACL heritage](02-fipa-acl-heritage/) | Construir | Estándar de comunicación de agentes. |
| 03 | [Communication protocols](03-communication-protocols/) | Construir | Request, inform, *call for proposals*. |
| 04 | [Primitive model](04-primitive-model/) | Construir | El modelo mínimo: creencias, deseos, intenciones. |

### Bloque 2 — Patrones arquitectónicos

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 05 | [Supervisor orchestrator pattern](05-supervisor-orchestrator-pattern/) | Construir | Un *manager* despachando a *workers*. |
| 06 | [Hierarchical architecture](06-hierarchical-architecture/) | Construir | Árboles de agentes con *scope* por nivel. |
| 07 | [Society of mind debate](07-society-of-mind-debate/) | Construir | Minsky: mente como sociedad de agentes. |
| 08 | [Role specialization](08-role-specialization/) | Construir | Roles especializados con CrewAI. |
| 09 | [Parallel swarm networks](09-parallel-swarm-networks/) | Construir | Múltiples agentes en paralelo. |
| 10 | [Group chat speaker selection](10-group-chat-speaker-selection/) | Construir | Quién habla en cada turno. |
| 11 | [Handoffs and routines](11-handoffs-and-routines/) | Construir | Transición controlada entre agentes. |

### Bloque 3 — Protocolos formales y coordinación

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 12 | [A2A protocol](12-a2a-protocol/) | Construir | Agent-to-Agent, *agent cards*, *tasks*. |
| 13 | [Shared memory blackboard](13-shared-memory-blackboard/) | Construir | Memoria compartida tipo *blackboard*. |
| 14 | [Consensus and BFT](14-consensus-and-bft/) | Construir | Consenso Bizantino entre agentes. |
| 15 | [Voting debate topology](15-voting-debate-topology/) | Construir | Votación y *debate* para robustez. |
| 16 | [Negotiation bargaining](16-negotiation-bargaining/) | Construir | *Contract Net*, subastas, bargaining. |
| 17 | [Generative agents simulation](17-generative-agents-simulation/) | Construir | Simuladores estilo *Smallville*. |
| 18 | [Theory of mind coordination](18-theory-of-mind-coordination/) | Construir | Modelos del otro agente. |

### Bloque 4 — Swarm optimization, MARL y producción

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 19 | [Swarm optimization PSO/ACO](19-swarm-optimization-pso-aco/) | Construir | Particle swarm, ant colony. |
| 20 | [MARL MADDPG/QMIX/MAPPO](20-marl-maddpg-qmix-mappo/) | Construir | Multi-agent reinforcement learning. |
| 21 | [Agent economies](21-agent-economies/) | Construir | Mercados entre agentes. |
| 22 | [Production scaling queues checkpoints](22-production-scaling-queues-checkpoints/) | Construir | Colas, *queues*, *checkpoints*. |
| 23 | [Failure modes MAST groupthink](23-failure-modes-mast-groupthink/) | Construir | MAST, *groupthink*, *cascading failures*. |
| 24 | [Evaluation coordination benchmarks](24-evaluation-coordination-benchmarks/) | Construir | Métricas de coordinación. |
| 25 | [Case studies 2026 SOTA](25-case-studies-2026-sota/) | Construir | MAGI, ChatDev, MetaGPT, AutoGen. |

## Prerrequisitos

- **Fases 14 y 15** completas.
- Conocimiento de LangGraph, AutoGen o CrewAI.
- Familiaridad con A2A y MCP (Fase 13).
- GPU con 16+ GB VRAM para las lecciones 20 (MARL).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Decidir** cuándo un sistema multi-agente es mejor que un
  agente con más capacidad (prompt + tool).
- **Implementar** los patrones clásicos: supervisor, jerárquico,
  swarm, handoffs, *blackboard*.
- **Construir** sistemas basados en A2A con agent cards y
  delegación de tareas.
- **Aplicar** mecanismos de consenso y votación para robustez.
- **Simular** sociedades de agentes estilo *Generative Agents*.
- **Escalar** sistemas multi-agente en producción con colas y
  *checkpoints*.
- **Diagnosticar** modos de falla: *groupthink*, *cascading
  failures*, *MAST*.
- **Conocer** los *frameworks* y casos de estudio SOTA 2026.

## Stack y herramientas

- **LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, Claude Agent
  SDK** para multi-agente.
- **A2A, MCP, FIPA-ACL** como protocolos.
- **PettingZoo, RLlib, VMAS** para MARL.
- **Ray, Temporal, DBOS** para producción.
- **NetworkX, Mesa** para simulaciones de agentes.
- **OpenTelemetry, Langfuse, Phoenix** para observabilidad.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **BDI** | Lección 04 | Modelo primitivo. |
| **Supervisor** | Lección 05 | Patrón dominante. |
| **Swarm** | Lección 09 | Paralelismo. |
| **A2A** | Lección 12 | Interoperabilidad. |
| **BFT** | Lección 14 | Consenso. |
| **Generative agents** | Lección 17 | Simulación. |
| **MARL** | Lección 20 | RL multi-agente. |
| **Groupthink** | Lección 23 | Falla. |

## Cómo estudiar esta fase

1. **Empieza por la lección 01 (why multi-agent).** Si no puedes
   defender con datos que el multi-agente aporta valor, no lo
   uses.
2. **Elige UN patrón y domínalo.** El supervisor (lección 05)
   cubre el 80% de los casos prácticos.
3. **A2A (lección 12) es el estándar emergente.** Si vas a
   interoperar entre frameworks, entiéndelo bien.
4. **Las lecciones 17 y 25 son las más "vistosas"** pero las más
   caras. Hazlas cuando tengas tiempo.
5. **Modos de falla (lección 23) es lectura obligada** antes de
   poner un multi-agente en producción.

## Verificación de progreso

```bash
# Lección 05 — supervisor orchestrator
python3 fases/16-multi-agente-y-enjambres/05-supervisor-orchestrator-pattern/code/main.py

# Lección 12 — A2A agent
python3 fases/16-multi-agente-y-enjambres/12-a2a-protocol/code/main.py

# Lección 17 — Generative agents
python3 fases/16-multi-agente-y-enjambres/17-generative-agents-simulation/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Topologías multi-agente

| Patrón | Cuándo usarlo | Lección |
|---|---|---|
| Supervisor | Coordinación central clara | 05 |
| Jerárquico | Equipos grandes | 06 |
| Swarm paralelo | Tareas independientes | 09 |
| Group chat | Brainstorming, debate | 10 |
| Handoffs | Triaje, escalación | 11 |
| Blackboard | Memoria compartida | 13 |
| Consenso | Decisiones críticas | 14 |
| Negociación | Recursos escasos | 16 |
| Teoría de la mente | Juegos cooperativos | 18 |

## Conexión con otras fases

- **Entrada** → [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md)
  y [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md).
- **Salida natural** → [Fase 17 — Infraestructura y producción](../17-infraestructura-y-produccion/README.md)
  y [Fase 19 — Proyectos capstone](../19-proyectos-capstone/README.md).
- **Reuso en** → Simulaciones, juegos, *game theory*, *agent
  economies*.

## Recursos recomendados

- *Multiagent Systems* — Wooldridge (PDF libre).
- *An Introduction to MultiAgent Systems* — Wooldridge, 2nd ed.
- *FIPA specifications* — <http://www.fipa.org>.
- *Generative Agents paper* — Park et al., 2023.
- *AutoGen* — Microsoft, 2024.
- *CrewAI* — <https://docs.crewai.com>.
- *Mesa — Agent-based modeling* — <https://mesa.readthedocs.io>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *multi-
  agent*, *supervisor*, *swarm*, *A2A*, *BFT*, *groupthink*.
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [Fase 19 — Proyectos capstone](../19-proyectos-capstone/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
