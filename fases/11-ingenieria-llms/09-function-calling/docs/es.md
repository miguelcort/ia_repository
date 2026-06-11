# Function calling

> Function calling en LLMs: define tools (name, description, parameters JSON Schema), LLM genera JSON {tool_name, arguments}, parse + validate (Pydantic), execute, return result. Standards: OpenAI Function Calling (parallel, streaming), Anthropic Tool Use (+Computer Use para screen nav), Cohere, Mistral, Llama 3+, Gemini. Parallel tool calls: múltiples en 1 response, asyncio.gather, +quality multi-tool. MCP (Model Context Protocol, Anthropic Nov 2024): JSON-RPC 2.0 standard USB-C para AI, 100+ servers (Slack, GitHub, Postgres, S3). Operator (OpenAI 2025): autonomous agent con tools. Frameworks: Vercel AI SDK, LangGraph, LlamaIndex, Composio (100+ integrations). SOTA 2024-25: parallel + MCP + agentic frameworks + reasoning + memory.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/03-outputs-estructurados
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar tool_to_schema.
- Implementar format_tool_call.
- Implementar parse_tool_calls.
- Implementar parallel_tool_calls.
- Diagnosticar MCP y frameworks.

## Constrúyelo

```python
def tool_to_schema(name, description, parameters):
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters}}
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
fase: 11
leccion: 09
---

1. Tool def JSON Schema.
2. LLM genera {tool, args}.
3. Parallel calls.
4. MCP standard.
5. Vercel AI SDK, LangGraph.
```

## Ejercicios

1. **Tool def**: definir 3 tools,
   ejecutar paralelo.
2. **MCP**: setup MCP server
   simple.
3. **Desafio**: agent
   + Computer Use.

## Lecturas recomendadas

- "Function Calling and Other API Updates" (OpenAI, 2023)
- "Introducing the Model Context Protocol" (Anthropic, 2024)
- "Tool Use (Claude)" (Anthropic, 2024)
- "Building Effective Agents" (Anthropic, 2024)
- "Composio: 100+ Tool Integrations" (2024)

---

> 📚 **Adaptación al español** de la lección "[Function Calling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).