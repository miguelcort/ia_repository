# 23 — Function call dispatcher

> Function call dispatcher: el LLM devuelve tool calls (name + args JSON), dispatcher valida schema, ejecuta, formatea response, devuelve al LLM. Frameworks: OpenAI function calling, Anthropic tool use, Google function calling.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 13, Fase 19/21, 19/22
**Tiempo estimado:** ~25 minutos

## Objetivos

- Parse LLM function call output.
- Validate args vs schema.
- Execute + format result.
- Return to LLM context.

## El problema

Function call dispatcher es el bridge entre LLM
output y tool execution. OpenAI format: `{"name":
"search", "arguments": '{"query": "..."}'}`. Anthropic
format: content blocks con `{"type": "tool_use",
"id": "...", "name": "...", "input": {...}}`. Dispatcher:
parse → validate (vs tool schema) → execute → format
result (OpenAI: `{"role": "tool", "content": "...",
"tool_call_id": "..."}`, Anthropic: `{"type":
"tool_result", "tool_use_id": "...", "content": "..."}`)
→ return al LLM. Loop hasta LLM returns final answer.

## Constrúyelo

```python
def dispatch_tool_call(tool_call, tool_registry):
    """Parse, validate, execute, format."""
    name = tool_call["name"]
    args = tool_call["arguments"]
    tool = tool_registry.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}
    try:
        validate(args, tool["schema"])
    except ValidationError as e:
        return {"error": f"Invalid args: {e}"}
    result = tool["fn"](**args)
    return {"result": str(result), "tool_call_id":
            tool_call.get("id")}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tool-dispatcher
fase: 19
leccion: 23
---

1. Parse function call.
2. Validate schema.
3. Execute.
4. Format response.
5. Return to LLM.
```

## Ejercicios

1. **OpenAI format**: dispatch
   5 tool calls.
2. **Anthropic format**: 5
   tool uses.
3. **Desafío**: parallel
   tool calls.

## Lecturas recomendadas

- "OpenAI Function Calling" (2023)
- "Anthropic Tool Use" (2024)
- "Google Function Calling" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[23-function-call-dispatcher]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
