"""
Lección: 06-mcp-fundamentals
Fase: 13
MCP (Model Context Protocol, Anthropic 2024): standard abierto
para conectar LLMs con herramientas, datos y sistemas. JSON-RPC.
Client + server architecture. Tools, Resources, Prompts.
"""
from __future__ import annotations
import json


def make_mcp_request(method, params=None, id_=1):
    """JSON-RPC 2.0 request."""
    req = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        req["params"] = params
    return req


def make_mcp_response(result=None, error=None, id_=1):
    """JSON-RPC 2.0 response."""
    resp = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result
    return resp


def mcp_initialize(client_info, capabilities=None):
    """MCP initialize handshake."""
    return make_mcp_request("initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": client_info,
        "capabilities": capabilities or {},
    })


def mcp_tools_list():
    """tools/list method."""
    return make_mcp_request("tools/list")


def mcp_tool_call(name, arguments):
    """tools/call method."""
    return make_mcp_request("tools/call", {
        "name": name,
        "arguments": arguments,
    })


def parse_mcp_message(raw):
    """Parse MCP message (JSON-RPC)."""
    return json.loads(raw)


def validate_initialize(result):
    """Validate initialize result."""
    if "serverInfo" not in result:
        return False
    if "protocolVersion" not in result:
        return False
    return True


def main() -> int:
    req = mcp_initialize({"name": "test-client", "version": "1.0.0"})
    print(json.dumps(req, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())