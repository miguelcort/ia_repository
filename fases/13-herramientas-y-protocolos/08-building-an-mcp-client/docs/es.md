# Building an MCP client

> MCP client: connect via transport (stdio/HTTP+SSE/WebSocket) + initialize handshake (clientInfo + capabilities) + list tools/resources/prompts + call tools (tools/call) + read resources (resources/read) + get prompts (prompts/get) + session management (connect, reconnect, close) + sync + async clients. Methods: connect()/close(), list_tools()/call_tool(), list_resources()/read_resource(), list_prompts()/get_prompt(), send_request() low-level JSON-RPC. Error handling: JSON-RPC error codes (-32700 parse, -32600 invalid request, -32601 method not found, -32602 invalid params, -32603 internal), try/except, reconnect on transport error, timeout, logging, retry, circuit breaker. +Insights: +standardized, +reusable, +interop, +modular, +robust, +reliable. Frameworks: mcp, fastmcp, anthropic, openai, langchain, asyncio, aiohttp. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE. Decision: scripting -> sync, server -> async, production -> async, web -> async. Trade-offs: sync + simple, async + concurrent. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native + ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MCPClient con connect/close.
- Implementar list_tools y call_tool.
- Implementar use_client.
- Diagnosticar sync vs async.
- Diagnosticar error handling MCP.

## Constrúyelo

```python
class MCPClient:
    async def connect(self):
        await asyncio.sleep(0.01)
        self.connected = True
        self.session_id = f"session_{id(self)}"
        return {"name": "mock-server", "version": "1.0.0"}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-client
fase: 13
leccion: 08
---

1. connect/close.
2. list/call.
3. Session management.
4. Sync + async.
5. +Reusable.
```

## Ejercicios

1. **MCP client**: construir
   client con mcp SDK.
2. **Async**: probar
   async client.
3. **Desafio**: client
   multi-server.

## Lecturas recomendadas

- "MCP Client SDK" (https://modelcontextprotocol.io)
- "Building MCP Clients" (Anthropic Cookbook, 2024)
- "MCP Error Handling" (Anthropic, 2024)
- "Async MCP Clients" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Building an MCP Client]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).