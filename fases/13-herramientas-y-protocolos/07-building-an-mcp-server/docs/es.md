# Building an MCP server

> Building MCP server: FastMCP (Pythonic) o mcp SDK. Componentes: Tools (name + description + params, @server.tool()), Resources (URI + content read-only, @server.resource()), Prompts (reusable templates, @server.prompt()). Schema auto-inferido de type hints, docstring -> description. JSON-RPC transports: (1) stdio (stdin/stdout, +simple -network), (2) HTTP+SSE (server-sent events, +network +streaming), (3) Streamable HTTP (+efficient +chunked), (4) WebSocket (+bidirectional +real-time +low latency). +Insights: +standardized, +reusable, +modular, +Pythonic, +auto-schema, +auto-validation, +auto-registration. Frameworks: mcp, fastmcp, langchain, anthropic, openai. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE. Decision: local -> stdio, server -> HTTP+SSE, real-time -> WebSocket, production -> HTTP+SSE o Streamable. Trade-offs: +simple -custom. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native + ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ToolRegistry con @tool decorator.
- Implementar ResourceRegistry con @resource decorator.
- Implementar PromptRegistry con @prompt decorator.
- Implementar build_mcp_server.
- Implementar handle_request.
- Diagnosticar transports MCP.

## Constrúyelo

```python
class ToolRegistry:
    def tool(self, name, description, parameters):
        def decorator(func):
            self.tools[name] = {"name": name, "description": description, "parameters": parameters, "func": func}
            return func
        return decorator
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-server
fase: 13
leccion: 07
---

1. @tool, @resource, @prompt.
2. JSON-RPC transport.
3. Stdio, HTTP+SSE.
4. +Auto-schema.
5. +Reusable.
```

## Ejercicios

1. **MCP server**: construir
   server con fastmcp.
2. **Resources**: agregar
   resources al server.
3. **Desafio**: custom
   transport MCP.

## Lecturas recomendadas

- "MCP Server SDK" (https://modelcontextprotocol.io)
- "FastMCP: Pythonic Servers" (Anthropic, 2024)
- "Building MCP Servers" (Anthropic Cookbook, 2024)
- "MCP Transports Specification" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Building an MCP Server]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).