"""
Lección: 12-mcp-roots-and-elicitation
Fase: 13
MCP roots: filesystem boundaries client provides a server.
Elicitation: server asks user structured input via client.
"""
from __future__ import annotations
import json


def make_roots_list(roots):
    """Build roots/list response."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "roots": [
                {"uri": r["uri"], "name": r.get("name", "")}
                for r in roots
            ],
        },
    }


def validate_root_uri(uri):
    """Validate root URI (file:// only)."""
    return uri.startswith("file://")


def within_root(path, root_uri):
    """Check if path is within root."""
    if not validate_root_uri(root_uri):
        return False
    # normalizar file:/// a file:// (quitar una barra)
    norm = lambda u: "file://" + u[len("file://"):].lstrip("/")
    p = norm(path)
    r = norm(root_uri)
    return p == r or p.startswith(r + "/")


def make_elicitation_request(message, schema):
    """Build elicitation request (server -> client)."""
    return {
        "jsonrpc": "2.0",
        "method": "elicitation/create",
        "params": {
            "message": message,
            "requestedSchema": schema,
        },
        "id": 1,
    }


def make_elicitation_response(values, action="accept"):
    """Build elicitation response (client -> server)."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "action": action,  # accept, decline, cancel
            "content": values,
        },
    }


def elicitation_actions():
    """Available elicitation actions."""
    return ["accept", "decline", "cancel"]


def main() -> int:
    roots = [{"uri": "file:///home/user/project", "name": "Project"}]
    r = make_roots_list(roots)
    print(json.dumps(r, indent=2))
    print(f"Within root: {within_root('file:///home/user/project/src/main.py', 'file:///home/user/project')}")
    eli = make_elicitation_request(
        "What's your favorite color?",
        {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]},
    )
    print(json.dumps(eli, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())