# Fase 15 — Sistemas autónomos

> Agentes que mejoran, se evalúan y se gobiernan a sí mismos.

Esta fase cierra el ciclo de los **sistemas autónomos**: agentes que
no solo ejecutan tareas, sino que **mejoran su propio código**,
ejecutan **investigación científica end-to-end**, navegan la web
por sí mismos, y se **auto-gobiernan** dentro de límites seguros.
Cubre la familia STAR, AlphaEvolve, Darwin Gödel Machine, AI
Scientist v2, recursive self-improvement, *coding agent landscape*,
Claude Code, browser agents, durable execution, *cost governors*,
*kill switches*, Constitutional AI, Llama Guard, RSP, OpenAI
Preparedness, METR y CAIS.

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–4) cubre el **razonamiento de horizonte largo**:
long-horizon agents, familia STAR, AlphaEvolve, Darwin Gödel
Machine. El **bloque 2** (5–8) entra a la **investigación
autónoma**: AI Scientist v2, automated alignment research,
recursive y bounded self-improvement. El **bloque 3** (9–16)
trata la **operación de coding agents**: landscape, Claude Code,
browser agents, durable execution, cost governors, kill switches,
propose-then-commit, checkpoints y rollback. El **bloque 4**
(17–22) cubre la **gobernanza y seguridad**: Constitutional AI,
Llama Guard, RSP, OpenAI Preparedness + DeepMind FSF, METR y
CAIS/CAISI.

## Índice de lecciones

### Bloque 1 — Razonamiento de horizonte largo

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Long-horizon agents](01-long-horizon-agents/) | Construir | Tareas de horas o días, *state durability*. |
| 02 | [STAR family reasoning](02-star-family-reasoning/) | Construir | STAR-1, STAR-2 y variantes. |
| 03 | [AlphaEvolve y evolutionary coding](03-alphaevolve-evolutionarycoding/) | Construir | Búsqueda evolutiva sobre código. |
| 04 | [Darwin Gödel Machine](04-darwin-godel-machine/) | Construir | Auto-mejora con verificación. |

### Bloque 2 — Investigación autónoma

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 05 | [AI Scientist v2](05-ai-scientist-v2/) | Construir | Hipótesis → experimento → paper. |
| 06 | [Automated alignment research](06-automated-alignment-research/) | Construir | Agentes investigando alineamiento. |
| 07 | [Recursive self-improvement](07-recursive-self-improvement/) | Construir | RSI con cuidado y métricas. |
| 08 | [Bounded self-improvement](08-bounded-self-improvement/) | Construir | RSI dentro de límites formales. |

### Bloque 3 — Operación de coding agents

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 09 | [Coding agent landscape](09-coding-agent-landscape/) | Construir | Claude Code, Cursor, OpenCode, Aider. |
| 10 | [Claude Code permission modes](10-claude-code-permission-modes/) | Construir | *Default*, *plan*, *accept-edits*, *bypass*. |
| 11 | [Browser agents](11-browser-agents/) | Construir | Chromium + Playwright + LLM. |
| 12 | [Durable execution](12-durable-execution/) | Construir | *Workflow engines*, *durable agents*. |
| 13 | [Cost governors](13-cost-governors/) | Construir | *Token budget*, *dollar budget*, *turn budget*. |
| 14 | [Kill switches y canaries](14-kill-switches-canaries/) | Construir | *Stop signals* y *deadman switches*. |
| 15 | [Propose-then-commit](15-propose-then-commit/) | Construir | PR con diff y plan. |
| 16 | [Checkpoints y rollback](16-checkpoints-rollback/) | Construir | Estado persistente, *undo*. |

### Bloque 4 — Gobernanza y seguridad

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 17 | [Constitutional AI](17-constitutional-ai/) | Construir | RLAIF, *critique-revise*. |
| 18 | [Llama Guard](18-llama-guard/) | Construir | Clasificador de seguridad. |
| 19 | [Anthropic RSP](19-anthropic-rsp/) | Construir | *Responsible Scaling Policy*. |
| 20 | [OpenAI Preparedness + DeepMind FSF](20-openai-preparedness-deepmind-fsf/) | Construir | *Frontier Safety Frameworks*. |
| 21 | [METR external evaluation](21-metr-external-evaluation/) | Construir | *Evals* independientes. |
| 22 | [CAIS / CAISI societal risk](22-cais-caisi-societal-risk/) | Construir | Center for AI Safety. |

## Prerrequisitos

- **Fases 13 y 14** completas.
- Conocimiento de LangGraph, AutoGen o frameworks similares.
- GPU con 24+ GB VRAM (recomendado para RSI con modelos grandes).
- Familiaridad con conceptos de seguridad y *governance*.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Implementar** un agente de horizonte largo con *state
  durability* y *checkpoints*.
- **Comparar** AlphaEvolve, Darwin Gödel Machine y AI Scientist
  v2 como paradigmas de auto-mejora.
- **Operar** un coding agent con permission modes, cost
  governors y kill switches.
- **Construir** un browser agent funcional con Playwright o
  Chromium DevTools.
- **Diseñar** un *constitutional AI* pipeline con critique-revise
  y *RLAIF*.
- **Aplicar** *Frontier Safety Frameworks* (RSP, OpenAI
  Preparedness, FSF) a un sistema de producción.
- **Evaluar** un sistema autónomo con METR-style *evals*
  externos.

## Stack y herramientas

- **LangGraph, AutoGen, CrewAI** para agentes.
- **Claude Code SDK, OpenCode, Aider, SWE-agent** para coding.
- **Playwright, Chromium DevTools, Browser Use** para browser
  agents.
- **Temporal, Restate, DBOS** para *durable execution*.
- **OpenTelemetry, Langfuse, Phoenix** para observabilidad.
- **Constitutional AI, Llama Guard, ShieldGemma** para
  seguridad.
- **METR, Apollo Research, CAIS** para evaluación externa.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Long-horizon** | Lección 01 | Cualquier agente serio. |
| **AlphaEvolve** | Lección 03 | Evolutionary coding. |
| **DGM** | Lección 04 | Auto-mejora verificable. |
| **AI Scientist** | Lección 05 | Investigación autónoma. |
| **RSI** | Lecciones 07, 08 | Self-improvement. |
| **Permission modes** | Lección 10 | Claude Code. |
| **Cost governor** | Lección 13 | Operación. |
| **RSP** | Lección 19 | Frontier safety. |
| **FSF** | Lección 20 | DeepMind. |

## Cómo estudiar esta fase

1. **La lección 01 (long-horizon) es la base conceptual.** Sin
   entender *state durability*, todo lo demás parece
   investigación teórica.
2. **Las lecciones 13, 14, 15, 16 son operacionalmente
   críticas.** Léelas aunque no vayas a implementar todas.
3. **El bloque 4 (gobernanza) es lectura obligada para
   deployment.** No es opcional si tu agente va a producción.
4. **No intentes implementar RSI a la ligera.** La lección 08
   (bounded RSI) es la única forma realista de hacerlo.
5. **Las *frameworks* cambian rápido.** Cíñete a los conceptos
   (durable execution, cost governor) más que a la
   implementación específica.

## Verificación de progreso

```bash
# Lección 01 — long-horizon agent
python3 fases/15-sistemas-autonomos/01-long-horizon-agents/code/main.py

# Lección 12 — durable execution
python3 fases/15-sistemas-autonomos/12-durable-execution/code/main.py

# Lección 17 — Constitutional AI
python3 fases/15-sistemas-autonomos/17-constitutional-ai/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## El ciclo de auto-mejora

```text
[agente v_n]  →  propón cambio  →  [verificación]  →  acept?
       ↑                                              ↓
       └──────────── rollback si falla ←─────────────┘
                                                       ↓
                                                  [agente v_{n+1}]
```

Este es el patrón de Darwin Gödel Machine (lección 04). Bounded
RSI (lección 08) limita el espacio de cambios permitidos.

## Conexión con otras fases

- **Entrada** → [Fase 13 — Herramientas y protocolos](../13-herramientas-y-protocolos/README.md)
  y [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- **Salida natural** → [Fase 16 — Multi-agente y enjambres](../16-multi-agente-y-enjambres/README.md)
  y [Fase 18 — Ética y alineación](../18-etica-y-alineacion/README.md).
- **Reuso en** → Fase 17 (producción), Fase 19 (capstone).

## Recursos recomendados

- *Darwin Gödel Machine paper* — Zhang et al., 2025.
- *AI Scientist v2* — Sakana AI, 2025.
- *Anthropic Responsible Scaling Policy* — Anthropic, 2024.
- *OpenAI Preparedness Framework* — OpenAI, 2023.
- *DeepMind Frontier Safety Framework* — Google DeepMind, 2024.
- *METR — Measuring AI Ability to Complete Long Tasks* — METR, 2025.
- *Claude Code documentation* — Anthropic.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *long-
  horizon agent*, *RSI*, *DGM*, *RSP*, *FSF*, *Llama Guard*.
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [Fase 18 — Ética y alineación](../18-etica-y-alineacion/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
