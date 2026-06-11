# Function calling deep dive

> Function calling deep dive: tool_choice (auto/required/none/specific), parallel_tool_calls (multiple tools en 1 turn, -latency, +throughput), strict mode (all properties required, additionalProperties=false, +validation, -hallucinations, +reliability), JSON mode / response_format (structured output sin tool). Best practices: +descriptive name (snake_case verbo specific), +clear description (cuando usar + cuando NO + examples), +JSON schema valid (required types enums constraints), +min parameters, +strict mode, +tool_choice, +error handling, +logging, +rate limit, +cost tracking. +Insights: +structured +validated +modular. Frameworks: openai, anthropic, langchain, vLLM, transformers, MCP, A2A. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP. Trade-offs: tools + safe, freeform + flexible, strict + safe, parallel + efficiency. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar tool_call_payload y assistant_message_with_tool_calls.
- Implementar tool_result_message (role=tool).
- Implementar conversation_with_tools.
- Implementar tool_choice (auto, required, none, specific).
- Implementar strict_mode_schema.
- Diagnosticar best practices.

## Constrúyelo

```python
def strict_mode_schema(schema):
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": list(schema.get("properties", {}).keys()),
        "additionalProperties": False,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: function-calling
fase: 13
leccion: 02
---

1. tool_choice 4 modes.
2. parallel_tool_calls.
3. Strict mode.
4. JSON mode.
5. Best practices.
```

## Ejercicios

1. **OpenAI tools**: usar
   strict mode con openai.
2. **Parallel**: probar
   parallel_tool_calls.
3. **Desafio**: custom
   tool router con retries.

## Lecturas recomendadas

- "OpenAI Function Calling Guide" (OpenAI, 2023)
- "Anthropic Tool Use" (Anthropic, 2024)
- "Parallel Function Calling" (OpenAI Cookbook, 2024)
- "Strict Mode for Tool Calls" (OpenAI Cookbook, 2024)

---

> 📚 **Adaptación al español de la lección [Function Calling Deep Dive]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).