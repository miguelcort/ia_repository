# 01 — Agente de coding nativo de terminal

> En 2026 la forma de un agente de coding está resuelta. Un harness TUI, un plan con estado, una superficie de herramientas con permisos, un loop que planifica, actúa, observa y se recupera. Claude Code, Cursor 3 y OpenCode se ven igual a 50 pies de distancia. Este capstone te pide construir uno de extremo a extremo — CLI a la entrada, pull request a la salida — y medirlo contra mini-swe-agent y Live-SWE-agent en SWE-bench Pro. Vas a aprender por qué la parte difícil no es la llamada al modelo sino el loop de herramientas, el sandbox y el techo de costo en una corrida de 50 turnos.

**Tipo:** Capstone
**Lenguajes:** TypeScript / Bun (harness), Python (scripts de eval)
**Prerrequisitos:** Fase 11 (Ingeniería de LLM), Fase 13 (Herramientas y protocolos), Fase 14 (Agentes), Fase 15 (Sistemas autónomos), Fase 17 (Infraestructura)
**Fases ejercitadas:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Tiempo estimado:** 35 horas

## Objetivos de aprendizaje

- Implementar el loop plan-act-observe-recover con presupuesto
  acotado (turnos, tokens, dólares).
- Construir una superficie de herramientas con sandbox y
  truncado de salida.
- Cablear los ocho eventos de hooks 2026
  (`PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`,
  `UserPromptSubmit`, `Notification`, `Stop`, `PreCompact`).
- Evaluar tu harness contra `mini-swe-agent` y `Live-SWE-agent`
  en SWE-bench Pro.
- Diagnosticar modos de falla comunes: tool-loop instability,
  *context poisoning*, *runaway cost* y operaciones destructivas
  de filesystem.

## El problema

Los agentes de coding se volvieron la categoría dominante de
aplicación de IA en 2026. Claude Code (Anthropic), Cursor 3 con
Composer 2 y Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode
(112k stars), Factory Droids y Google Jules envían variaciones de
la misma arquitectura: un harness de terminal, una superficie de
herramientas con permisos, un sandbox, y un loop
plan-act-observe construido alrededor de un modelo frontera. La
frontera es estrecha — Live-SWE-agent alcanzó 79.2% en SWE-bench
Verified con Opus 4.5 — pero la ingeniería es amplia. La mayoría
de los modos de falla no son errores del modelo. Son
inestabilidad del tool-loop, envenenamiento de contexto, costo
desbordado y operaciones destructivas de filesystem.

No puedes razonar sobre estos agentes desde afuera. Tienes que
construir uno, ver el loop romperse en el turno 47 cuando ripgrep
devuelve 8MB de matches, y reconstruir la capa de truncado. Ese
es el punto de este capstone.

## El concepto

El harness tiene cuatro superficies. **Plan** mantiene un objeto
de estado estilo TodoWrite que el modelo reescribe cada turno.
**Act** despacha llamadas a herramientas (read, edit, run, search,
git). **Observe** captura stdout / stderr / exit codes, trunca, y
devuelve el resumen al modelo. **Recover** maneja errores de
herramientas sin volar la ventana de contexto ni loopear por
siempre. La forma 2026 añade una cosa más: **hooks**. `PreToolUse`,
`PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`,
`Notification`, `Stop` y `PreCompact` — puntos de extensión
configurables donde el operador inyecta política, telemetría y
*guardrails*.

El sandbox es E2B o Daytona. Cada tarea corre en un *devcontainer*
fresco con un *worktree* de git montado lectura-escritura. El
harness nunca toca el filesystem del host. El worktree se
desmonta en éxito o falla. El control de costo se aplica en tres
capas: un techo de tokens por turno, un presupuesto de dólares
por sesión, y un límite duro de turnos (típicamente 50). La capa
de observabilidad es OpenTelemetry con *spans* de las
convenciones semánticas GenAI, enviados a un Langfuse
*self-hosted*.

## Arquitectura

```text
  CLI del usuario  ->  harness (Bun + Ink TUI)
                            |
                            v
              loop plan / act / observe <-> Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                            |            (vía OpenRouter, model-agnostic)
                            v
              despachador de tools (cliente MCP StreamableHTTP)
                            |
       +-------------------+----------+--------+
       v          v             v         v
    read/edit   ripgrep    tree-sitter  git/run
       |          |             |         |
       +-------------------+----------+--------+
                            |
                            v
              sandbox E2B / Daytona (worktree aislado)
                            |
                            v
              hooks: Pre/Post, Session, Prompt, Compact
                            |
                            v
              OpenTelemetry -> Langfuse (spans, tokens, $)
                            |
                            v
              PR via GitHub app
```

## Stack

- **Runtime del harness:** Bun 1.2 + Ink 5 (React-in-terminal).
- **Acceso a modelos:** API unificada OpenRouter con Claude
  Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (para las
  tareas más difíciles).
- **Transporte de tools:** Model Context Protocol StreamableHTTP
  (revisión MCP 2026).
- **Sandbox:** sandboxes E2B (JS SDK) o devcontainers de Daytona.
- **Búsqueda de código:** subprocess de ripgrep, parsers
  tree-sitter para 17 lenguajes (pre-compilados).
- **Aislamiento:** `git worktree add` por tarea, limpieza en
  éxito / falla.
- **Eval harness:** SWE-bench Pro (subset verificado) +
  Terminal-Bench 2.0 + tu propio holdout de 30 tareas.
- **Observabilidad:** SDK OpenTelemetry con semconv `gen_ai.*` →
  Langfuse *self-hosted*.
- **PR posting:** GitHub App con token de *fine-grained*, scope
  limitado al repo objetivo.

## Constrúyelo

1. **TUI y command loop.** Scaffolding de un proyecto Bun con
   Ink. Acepta `agent run <repo> "<task>"`. Imprime una vista
   dividida: panel de plan (arriba), stream de tool calls
   (medio), presupuesto de tokens (abajo). Agrega cancel con
   Ctrl-C que dispara el hook `SessionEnd` antes de salir.

2. **Estado del plan.** Define un schema tipado TodoWrite
   (pending / in_progress / done con notas). El modelo reescribe
   el estado completo cada turno como tool call — no le
   permitas mutar incrementalmente. Persiste el plan a
   `.agent/state.json` para que los crashes puedan reanudar.

3. **Superficie de tools.** Define seis tools: `read_file`,
   `edit_file` (con preview de diff), `ripgrep`,
   `tree_sitter_symbols`, `run_shell` (con timeout), `git`
   (status / diff / commit / push). Expón sobre MCP
   StreamableHTTP para que el harness sea agnóstico al
   transporte. Cada tool devuelve salida truncada (cap a 4k
   tokens por call).

4. **Wrapping del sandbox.** Cada tarea genera un sandbox E2B.
   `git worktree add -b agent/$TASK_ID` una rama fresca. Todas
   las tool calls ejecutan dentro del sandbox. El filesystem del
   host es inalcanzable.

5. **Hooks.** Implementa los ocho tipos de hooks 2026. Cablea
   al menos cuatro hooks definidos por el usuario: (a)
   `PreToolUse` guard de comandos destructivos que bloquea
   `rm -rf` fuera del worktree, (b) `PostToolUse` contabilidad
   de tokens, (c) `SessionStart` inicialización de presupuesto,
   (d) `Stop` escribe un bundle final de trazas.

6. **Loop de eval.** Clona un subset de 30 issues de SWE-bench
   Pro Python. Corre tu harness contra cada uno. Compara contra
   `mini-swe-agent` (la baseline mínima) en pass@1,
   turns-per-task y $-per-task. Escribe los resultados a
   `eval/results.jsonl`.

7. **Control de costo.** Cortes duros: 50 turnos, 200k contexto,
   $5 por tarea. El hook `PreCompact` resume turnos más viejos a
   un bloque de estado previo en la marca 150k, liberando
   espacio para nuevas observaciones sin perder el plan.

8. **PR posting.** En éxito, el paso final es `git push` + una
   llamada a la API de GitHub que abre un PR con el plan y el
   resumen del diff en el body.

## Úsalo

```bash
$ agent run ./mi-repo "Arregla la race condition en worker.rs"
[plan]  1 localizar worker.rs y enumerar usos de mutex
        2 identificar estado compartido bajo contención
        3 proponer fix, verificar tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Despliégalo

El *skill* entregable vive en `outputs/skill-terminal-coding-agent.md`.
Dado un path de repo y una descripción de tarea, corre el loop
plan-act-observe completo en un sandbox y devuelve una URL de PR
más un bundle de trazas. La rúbrica de este capstone:

| Peso | Criterio | Cómo se mide |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Tu harness vs `mini-swe-agent` en 30 tareas Python emparejadas. |
| 20 | Claridad de arquitectura | Separación plan/act/observe, superficie de hooks, schema de tools — revisado contra el layout de `Live-SWE-agent`. |
| 20 | Seguridad | Tests de escape de sandbox, prompts de permisos, el guard de comandos destructivos pasa *red-team*. |
| 20 | Observabilidad | Completitud de trazas (100% de tool calls *spanned*), contabilidad de tokens por turno. |
| 15 | UX del desarrollador | Cold-start < 2s, recuperación de crash reanuda el plan, Ctrl-C cancela a mitad de tool limpiamente. |
| **100** | | |

## Ejercicios

1. Intercambia el modelo de respaldo de Claude Sonnet 4.7 a
   Qwen3-Coder-30B servido en vLLM. Compara pass@1 y
   $-per-task. Reporta dónde el modelo open queda atrás.
2. Añade un sub-agente `reviewer` que lea el diff antes de
   postear el PR y pueda pedir un loop de revisión. Mide si
   las revisiones *false-positive* bajan la tasa de pass de
   SWE-bench por debajo de la baseline de un solo agente
   (pista: usualmente sí).
3. Stress-test del sandbox: escribe una tarea que intente
   `curl` una URL externa y una tarea que escriba fuera del
   worktree. Confirma que ambas son bloqueadas por el hook
   `PreToolUse`. Loguea los intentos.
4. Implementa la sumarización de `PreCompact` con un modelo más
   pequeño (Haiku 4.5). Mide cuánta fidelidad del plan se
   pierde a 3x compactación.
5. Intercambia el transporte MCP StreamableHTTP por stdio.
   Mide cold-start y latencia por call. Escoge un ganador para
   uso local-only.

## Términos clave

| Término | Lo que dice la gente | Lo que realmente significa |
|---|---|---|
| **Harness** | "El loop del agente" | El código que rodea al modelo y despacha tools, mantiene el estado del plan y aplica presupuestos. |
| **Hook** | "Listener de eventos del agente" | Un script definido por el usuario que corre en uno de ocho eventos de lifecycle disparados por el harness. |
| **Worktree** | "Sandbox de git" | Un checkout de git enlazado en un path separado; descartable sin tocar el clon principal. |
| **TodoWrite** | "Estado del plan" | Una lista tipada de items pending/in-progress/done que el modelo reescribe cada turno. |
| **StreamableHTTP** | "Transporte MCP" | Revisión MCP 2026: conexión HTTP de larga duración con streaming bidireccional; reemplaza a SSE. |
| **Token ceiling** | "Presupuesto de contexto" | Cap por turno o por sesión de tokens de input+output; dispara compactación o terminación. |
| **pass@1** | "Tasa de pass en un solo intento" | Fracción de tareas SWE-bench resueltas en la primera corrida sin retry ni peek del test set. |

## Lecturas recomendadas

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — harness de referencia de Anthropic.
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs y Composer 2 notas de producto.
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — baseline mínima para comparación de harness en SWE-bench.
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified con Opus 4.5.
- [OpenCode](https://opencode.ai) — harness open, 112k stars.
- [SWE-bench Pro leaderboard](https://www.swebench.com) — la evaluación objetivo de este capstone.
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata.
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schema de spans para tool calls y uso de tokens.

---

> 📚 **Adaptación al español** de la lección
> "[Capstone 01 — Terminal-Native Coding Agent]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
