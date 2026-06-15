# 13 — MCP server with registry

> MCP (Model Context Protocol, Anthropic 2024): standard para tools, resources, prompts en LLM apps. Servers exponen capabilities, clients consumen via JSON-RPC. Registry para discovery (Docker Hub, Smithery). Capstone: build server + registry.

**Tipo:** Capstone
**Lenguajes:** Python (server), TypeScript (client)
**Prerrequisitos:** Fase 13 (tools), Fase 11
**Tiempo estimado:** 20 horas

## Objetivos

- Implementar MCP server con tools + resources.
- JSON-RPC over stdio/HTTP.
- Registry para discovery.
- Eval sobre MCP test suite.

## El problema

MCP (Model Context Protocol) es el standard de Anthropic
para tools, resources, prompts en LLM apps. Server
expone: (1) Tools: functions callable. (2) Resources:
data files, DB rows. (3) Prompts: templates. Cliente
(MCP client en Claude, Cursor, OpenCode) descubre
y consume. Transport: stdio (local) o HTTP+SSE
(remote). Registry (Docker Hub, Smithery, Smithery)
permite discovery. Build: server con N tools, registry
para discovery, cliente con auto-discovery.

## Constrúyelo

```python
from mcp.server import Server, stdio
from mcp.types import Tool


app = Server("my-server")


@app.list_tools()
async def list_tools():
    return [Tool(name="search",
                description="Search docs",
                inputSchema={...})]


@app.call_tool()
async def call_tool(name, arguments):
    if name == "search":
        return search_docs(arguments["query"])


if __name__ == "__main__":
    stdio.run(app)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mcp-server
fase: 19
leccion: 13
---

1. MCP server con tools.
2. JSON-RPC over stdio.
3. Registry (Smithery).
4. Client integration.
```

## Ejercicios

1. **MCP server**: tool de
   búsqueda sobre 10K docs.
2. **Registry**: publicar en
   Smithery.
3. **Desafío**: server con
   resources + prompts.

## Lecturas recomendadas

- "MCP Specification" (Anthropic 2024)
- "Smithery Registry" (2024)
- "MCP Python SDK" (Anthropic)

---

> 📚 **Adaptación al español** de la lección
> "[13-mcp-server-with-registry]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
