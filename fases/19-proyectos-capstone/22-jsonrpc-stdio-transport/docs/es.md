# 22 — JSON-RPC stdio transport

> JSON-RPC over stdio: transporte principal para MCP local. Simple, language-agnostic, line-delimited JSON. Frames: request, response, notification. Methods: initialize, tools/list, tools/call. Errors: parse error, invalid request, method not found.

**Tipo:** Construir
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 13, Fase 19/21
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar JSON-RPC 2.0 server.
- Stdio transport (line-delimited).
- Request/response correlation.
- Error handling.

## El problema

JSON-RPC 2.0 sobre stdio es el transporte estándar
para MCP local servers. Cliente escribe JSON line
(stdout), server lee (stdin), escribe response.
Frame format: `{"jsonrpc": "2.0", "id": 1, "method":
"tools/list", "params": {}}`. Response: `{"jsonrpc":
"2.0", "id": 1, "result": {...}}`. Error: `{"jsonrpc":
"2.0", "id": 1, "error": {"code": -32600, "message":
"..."}}`. Notification: sin id (fire-and-forget).

## Constrúyelo

```python
import sys
import json


def jsonrpc_server(handlers):
    """JSON-RPC 2.0 over stdio."""
    for line in sys.stdin:
        req = json.loads(line)
        method = req.get("method")
        req_id = req.get("id")
        if method not in handlers:
            send_error(req_id, -32601, "Method not found")
            continue
        try:
            result = handlers[method](req.get("params", {}))
            if req_id is not None:
                send_response(req_id, result)
        except Exception as e:
            send_error(req_id, -32603, str(e))


def send_response(req_id, result):
    print(json.dumps({"jsonrpc": "2.0", "id": req_id,
                     "result": result}), flush=True)


def send_error(req_id, code, message):
    print(json.dumps({"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": code, "message": message}}),
          flush=True)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-jsonrpc
fase: 19
leccion: 22
---

1. JSON-RPC 2.0 over stdio.
2. Request/response/notification.
3. Error codes.
4. Frame parsing.
5. MCP-compatible.
```

## Ejercicios

1. **Server**: 3 methods.
2. **Client**: send + receive.
3. **Desafío**: error
   handling + timeouts.

## Lecturas recomendadas

- "JSON-RPC 2.0" (Spec 2010)
- "MCP stdio transport" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[22-jsonrpc-stdio-transport]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
