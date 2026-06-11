"""
Lección: 14-model-context-protocol
Fase: 11
MCP (Model Context Protocol, Anthropic 2024): standard USB-C para AI.
Resources, tools, prompts, JSON-RPC.
"""
from __future__ import annotations
import json
import sys
import numpy as np


def mcp_request(method, params=None, id=1):
    """JSON-RPC 2.0 request format."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params or {},
    }


def mcp_response(result, id=1):
    """JSON-RPC 2.0 response format."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


def mcp_error(code, message, id=1):
    """JSON-RPC 2.0 error format."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message},
    }


def mcp_tool_def(name, description, params):
    """Define un MCP tool."""
    return {
        "name": name,
        "description": description,
        "inputSchema": params,
    }


def mcp_components():
    """MCP protocol components."""
    return {
        "Resources": "Data sources (files, DBs, APIs). URI + content.",
        "Tools": "Actions the LLM can call. JSON Schema.",
        "Prompts": "Reusable prompt templates.",
        "Transport": "stdio, HTTP, WebSocket.",
    }


def main() -> int:
    req = mcp_request("tools/list", {}, id=1)
    print(f"Request: {json.dumps(req, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())