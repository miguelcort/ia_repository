# MCP fundamentals

> MCP (Anthropic 2024, Model Context Protocol): standard abierto para conectar LLMs con herramientas, datos, sistemas via JSON-RPC 2.0 + client-server architecture. Componentes: (1) Tools: ejecutan acciones (name + description + params, tools/list + tools/call), (2) Resources: data exposure read-only (URI scheme, resources/list + resources/read), (3) Prompts: reusable templates (prompts/list + prompts/get), (4) Sampling: server asks client LLM, (5) Roots: filesystem roots client-provided, (6) Elicitation: ask user structured input, (7) Async tasks: long-running. Initialize handshake: client -> server (initialize con clientInfo + protocolVersion 2024-11-05 + capabilities) -> server response (serverInfo + protocolVersion + capabilities). +Variants: Anthropic official, open source, FastMCP, MCP for Python. Frameworks: mcp, fastmcp, langchain, anthropic, openai. +Insights: +standardized, +modular, +reusable, +interop. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE, multi-agent. Trade-offs: standard + interop, custom + simple, function calling + simple. 2025: +MCP + A2A + native + ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01, 13/05
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_mcp_request y make_mcp_response (JSON-RPC).
- Implementar mcp_initialize handshake.
- Implementar mcp_tools_list y mcp_tool_call.
- Implementar parse_mcp_message y validate_initialize.
- Diagnosticar arquitectura MCP.

## Constrúyelo

```python
def make_mcp_request(method, params=None, id_=1):
    req = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        req["params"] = params
    return req
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-fundamentals
fase: 13
leccion: 06
---

1. JSON-RPC 2.0.
2. Client + server.
3. Tools, Resources, Prompts.
4. Initialize handshake.
5. +SOTA 2024-25.
```

## Ejercicios

1. **MCP server**: construir
   MCP server con fastmcp.
2. **MCP client**: usar
   MCP client Python.
3. **Desafio**: custom
   MCP server.

## Lecturas recomendadas

- "Model Context Protocol Specification" (Anthropic, 2024)
- "MCP Python SDK" (https://github.com/modelcontextprotocol/python-sdk)
- "FastMCP: Pythonic MCP Servers" (Anthropic, 2024)
- "Building MCP Servers" (Anthropic Cookbook, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Fundamentals]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).