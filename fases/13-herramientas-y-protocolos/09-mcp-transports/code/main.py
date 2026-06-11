"""
Lección: 09-mcp-transports
Fase: 13
MCP transports deep dive: stdio, HTTP+SSE, Streamable HTTP.
JSON-RPC encoding. Headers, auth, streaming.
"""
from __future__ import annotations
import json


def stdio_transport_encode(message):
    """Encode message para stdio transport (JSON + newline)."""
    return json.dumps(message) + "\n"


def stdio_transport_decode(line):
    """Decode stdio transport line."""
    return json.loads(line)


def http_sse_event(data, event_id=None, event_type="message"):
    """Encode SSE event."""
    s = ""
    if event_id is not None:
        s += f"id: {event_id}\n"
    s += f"event: {event_type}\n"
    s += f"data: {json.dumps(data)}\n\n"
    return s


def http_sse_parse_chunk(chunk):
    """Parse SSE chunk."""
    events = []
    current = {}
    for line in chunk.split("\n"):
        if line.startswith("data: "):
            current["data"] = json.loads(line[6:])
        elif line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def streamable_http_chunk(message):
    """Encode chunk para Streamable HTTP (NDJSON)."""
    return json.dumps(message) + "\n"


def transport_stdio():
    return {"name": "stdio", "bidirectional": True, "streaming": False, "network": False}


def transport_http_sse():
    return {"name": "http_sse", "bidirectional": False, "streaming": True, "network": True}


def transport_streamable_http():
    return {"name": "streamable_http", "bidirectional": True, "streaming": True, "network": True}


def transport_websocket():
    return {"name": "websocket", "bidirectional": True, "streaming": True, "network": True}


def main() -> int:
    msg = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    encoded = stdio_transport_encode(msg)
    print(f"stdio encoded: {encoded!r}")
    sse = http_sse_event(msg, event_id="1")
    print(f"SSE event:\n{sse}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())