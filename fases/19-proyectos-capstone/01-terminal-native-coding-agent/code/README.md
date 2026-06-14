# 01 — Terminal-native coding agent

Ejecuta el demo del loop plan-act-observe (sin llamadas de red):

```bash
cd code
python3 main.py
```

Deberías ver:

- El plan TodoWrite reescribiéndose cada turno.
- El conteo de turns, tokens y dólares consumidos.
- La traza de eventos de hooks.

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

Las 8 clases de tests cubren: TodoItem, PlanState, Budget,
HookBus, destructive_guard, tools (read_file, run_shell, escape
de path, truncado), run_agent y main.

## Para producción

Este demo usa `main.py` con un *modelo stub* determinista. Para
usar el capstone con un LLM real, debes:

1. Reemplazar `model_step` con una llamada a OpenRouter
   (Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro).
2. Envolver las tool calls en un sandbox E2B.
3. Añadir hooks reales (pre/post tool use, session lifecycle).
4. Configurar OpenTelemetry → Langfuse para observabilidad.
