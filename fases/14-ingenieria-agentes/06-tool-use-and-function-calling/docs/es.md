# Tool use and function calling

> Tool use y function calling en agent context: (1) function calling con JSON schemas, (2) OpenAI tools (tools + tool_calls) + Anthropic tool_use blocks + Google function calling, (3) parallel calls, (4) error handling + retries (exponential backoff, circuit breaker, fallback), (5) +Production reliable. Variants: OpenAI (standard +structured), Anthropic (flexible +tool_use blocks), Google (standard +function calling), MCP (standardized +reusable +cross-vendor), custom, Cohere, Mistral. +Structured, +Reliable, +Resilient, +Production, +Validated. Frameworks: openai, anthropic, langchain, MCP, google, vertex. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP. Decision: OpenAI -> standard, Anthropic -> flexible, Google -> standard, MCP -> cross-vendor, production -> MCP + Anthropic. Trade-offs: cada uno + specialty, structured + safe, freeform + simple. 2025: +MCP + A2A + native + formats.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01, 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar build_tool_def.
- Implementar parse_tool_call.
- Implementar execute_tool_call con retry.
- Implementar format_tool_result con truncation.
- Diagnosticar OpenAI vs Anthropic vs Google vs MCP.

## Constrúyelo

```python
def execute_tool_call(call_str, tool_registry, max_retries=3):
    name, args = parse_tool_call(call_str)
    for attempt in range(max_retries):
        if name in tool_registry:
            try:
                return tool_registry[name](**args)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"error after {max_retries} retries: {e}"
                time.sleep(0.01)
        else:
            return f"unknown tool: {name}"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: tool-use-agent
fase: 14
leccion: 06
---

1. Function calling.
2. JSON schemas.
3. OpenAI / Anthropic.
4. Parallel + retry.
5. +Production.
```

## Ejercicios

1. **OpenAI tools**: usar
   function calling con
   openai SDK.
2. **Anthropic**: probar
   tool_use blocks.
3. **Desafio**: agent
   production con retry.

## Lecturas recomendadas

- "OpenAI Function Calling" (OpenAI, 2024)
- "Anthropic Tool Use" (Anthropic, 2024)
- "MCP Tools" (Anthropic, 2024)
- "LangChain Tools" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Tool Use and Function Calling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).