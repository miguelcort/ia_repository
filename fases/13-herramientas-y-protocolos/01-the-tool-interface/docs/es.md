# The tool interface

> Tool interface: contrato entre LLM y herramientas externas. Tool spec = name (string identifier) + description (string cuando usar) + parameters (JSON schema: type=object, properties={name: {type, description, enum}}, required) + returns (output type). +Flow: LLM recibe tools en prompt -> LLM decide cuan llamar -> LLM genera JSON -> framework ejecuta -> result al LLM. +Variants: OpenAI function calling (tools=[{type: function, function: {name, description, parameters}}], tool_calls=[{id, function: {name, arguments}}], role=tool con tool_call_id), Anthropic tool use (tools=[{name, description, input_schema}], content=[{type: tool_use, id, name, input}], tool_result blocks), Google/Gemini function calling, LangChain tools. Frameworks: openai, anthropic, langchain, vLLM, transformers. +Production: standard en LLM apps 2024-25. +Use cases: API call, DB query, code exec, file ops, web search. Trade-offs: structured + safe, freeform + flexible. 2025: +Native + reasoning + MCP + A2A.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/09
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_tool spec.
- Implementar validate_parameters contra JSON schema.
- Implementar serialize_tool (OpenAI format).
- Implementar tool_router.
- Diagnosticar OpenAI vs Anthropic.

## Constrúyelo

```python
def make_tool(name, description, parameters, returns=None):
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "returns": returns or {"type": "string"},
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
name: tool-interface
fase: 13
leccion: 01
---

1. name + description + params.
2. JSON schema validation.
3. OpenAI tool_calls format.
4. Anthropic tool_use format.
5. +Modular +Validated.
```

## Ejercicios

1. **OpenAI tools**: usar
   OpenAI function calling
   con openai.
2. **Anthropic tools**:
   probar Anthropic tool use.
3. **Desafio**: custom
   tool router.

## Lecturas recomendadas

- "Function Calling" (OpenAI Documentation, 2023)
- "Tool Use" (Anthropic Documentation, 2024)
- "JSON Schema Specification" (https://json-schema.org)
- "LangChain Tools" (https://python.langchain.com/docs/modules/tools)

---

> 📚 **Adaptación al español de la lección [The Tool Interface]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).