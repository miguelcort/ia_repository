# Model Context Protocol (MCP)

> MCP (Model Context Protocol, Anthropic Nov 2024): standard USB-C para AI, JSON-RPC 2.0, transport stdio/HTTP/WebSocket. Componentes: Resources (URI-based data sources), Tools (JSON Schema actions), Prompts (reusable templates), Sampling (server-side LLM completions), Roots (filesystem boundaries). Frameworks: FastMCP (Python decorator-based), Anthropic SDK (TypeScript, Python), @modelcontextprotocol/sdk. Registries: glama.ai, mcp.run, smithery, Anthropic directory. +100 MCP servers en production: Slack, GitHub, Postgres, S3, Brave, Filesystem, Puppeteer, etc. Claude Desktop native support. LlamaIndex MCP loader, LangChain MCP adapters, Vercel AI SDK MCP. SOTA 2024-25: MCP standard + +100 servers + registries + multi-framework support + security (auth, scopes). Frontier: agentic MCP, multimodal (vision, audio), registries ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/09-function-calling
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MCP request/response.
- Implementar MCP tool def.
- Diagnosticar components y transport.
- Diagnosticar registries y frameworks.

## Constrúyelo

```python
def mcp_request(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp
fase: 11
leccion: 14
---

1. MCP JSON-RPC 2.0.
2. Resources, Tools, Prompts.
3. FastMCP, Anthropic SDK.
4. Transport stdio, HTTP, WebSocket.
5. glama.ai, mcp.run.
```

## Ejercicios

1. **MCP server**: implementar
   FastMCP server.
2. **MCP client**: integrar
   con Claude Desktop.
3. **Desafio**: MCP server
   custom con auth.

## Lecturas Recomendadas

- "Introducing the Model Context Protocol" (Anthropic, 2024)
- "MCP Specification" (Model Context Protocol, 2024)
- "FastMCP Documentation" (Anthropic, 2024)
- "Building MCP Servers" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Model Context Protocol]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).