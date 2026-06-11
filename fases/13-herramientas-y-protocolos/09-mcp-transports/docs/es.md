# MCP transports

> MCP transports deep dive: (1) stdio (stdin/stdout, JSON + newline, +simple, -network, local), (2) HTTP+SSE (Client -> Server: HTTP POST JSON-RPC, Server -> Client: SSE stream Content-Type: text/event-stream, event format: id + event + data + double newline, +HTTP, +streaming, -bidirectional, +chunked, browser), (3) Streamable HTTP (+Bidirectional, +Streaming, +HTTP, +Efficient, +Chunked NDJSON, -SSE, server), (4) WebSocket (+Bidirectional, +Real-time, +Low latency, +Production). +Insights: stdio + simple, HTTP + network, WebSocket + bidirectional, Streamable + efficient. Decision: local -> stdio, browser -> HTTP+SSE, server -> Streamable HTTP, real-time -> WebSocket, production -> HTTP+SSE o Streamable. Frameworks: mcp, fastmcp, anthropic, openai, langchain, asyncio. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE, web, server, real-time. Hoy: SOTA 2024-25 mix. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar stdio_transport_encode/decode.
- Implementar http_sse_event y http_sse_parse_chunk.
- Implementar streamable_http_chunk.
- Implementar transport metadata.
- Diagnosticar trade-offs stdio vs HTTP+SSE vs Streamable vs WebSocket.

## Constrúyelo

```python
def http_sse_event(data, event_id=None, event_type="message"):
    s = ""
    if event_id is not None:
        s += f"id: {event_id}\n"
    s += f"event: {event_type}\n"
    s += f"data: {json.dumps(data)}\n\n"
    return s
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-transports
fase: 13
leccion: 09
---

1. stdio.
2. HTTP+SSE.
3. Streamable HTTP.
4. WebSocket.
5. +Production.
```

## Ejercicios

1. **stdio**: construir
   MCP server con stdio.
2. **HTTP+SSE**: probar
   MCP server con HTTP+SSE.
3. **Desafio**: custom
   transport.

## Lecturas recomendadas

- "MCP Transports Specification" (Anthropic, 2024)
- "Server-Sent Events" (HTML Living Standard, 2024)
- "Streamable HTTP" (Anthropic, 2024)
- "WebSocket Protocol" (RFC 6455, 2011)

---

> 📚 **Adaptación al español de la lección [MCP Transports]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).