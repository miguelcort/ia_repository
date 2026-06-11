"""
Lección: 07-building-an-mcp-server
Fase: 13
Building an MCP server. FastMCP, decorators, tools/resources/prompts.
JSON-RPC transport. Stdio y HTTP+SSE.
"""
from __future__ import annotations
import json


class ToolRegistry:
    """Mock MCP tool registry."""
    def __init__(self):
        self.tools = {}

    def tool(self, name, description, parameters):
        """Decorator para registrar tool."""
        def decorator(func):
            self.tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters,
                "func": func,
            }
            return func
        return decorator

    def list_tools(self):
        """tools/list result."""
        return [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
                for t in self.tools.values()]

    def call(self, name, arguments):
        """tools/call dispatch."""
        if name not in self.tools:
            return {"error": f"tool not found: {name}"}
        try:
            return {"result": self.tools[name]["func"](**arguments)}
        except Exception as e:
            return {"error": str(e)}


class ResourceRegistry:
    """Mock MCP resource registry."""
    def __init__(self):
        self.resources = {}

    def resource(self, uri, description, mime_type="text/plain"):
        def decorator(func):
            self.resources[uri] = {
                "uri": uri,
                "description": description,
                "mimeType": mime_type,
                "func": func,
            }
            return func
        return decorator

    def list_resources(self):
        return [{"uri": r["uri"], "description": r["description"], "mimeType": r["mimeType"]}
                for r in self.resources.values()]

    def read(self, uri):
        if uri not in self.resources:
            return {"error": f"resource not found: {uri}"}
        return {"contents": self.resources[uri]["func"]()}


class PromptRegistry:
    """Mock MCP prompt registry."""
    def __init__(self):
        self.prompts = {}

    def prompt(self, name, description, arguments=None):
        def decorator(func):
            self.prompts[name] = {
                "name": name,
                "description": description,
                "arguments": arguments or [],
                "func": func,
            }
            return func
        return decorator

    def list_prompts(self):
        return [{"name": p["name"], "description": p["description"], "arguments": p["arguments"]}
                for p in self.prompts.values()]

    def get(self, name, args=None):
        if name not in self.prompts:
            return {"error": f"prompt not found: {name}"}
        return {"messages": self.prompts[name]["func"](**(args or {}))}


def build_mcp_server(name, version="1.0.0"):
    """Build MCP server with registries."""
    return {
        "name": name,
        "version": version,
        "tools": ToolRegistry(),
        "resources": ResourceRegistry(),
        "prompts": PromptRegistry(),
        "protocolVersion": "2024-11-05",
    }


def handle_request(server, request):
    """Handle MCP request."""
    method = request.get("method")
    params = request.get("params", {})
    if method == "tools/list":
        return {"result": {"tools": server["tools"].list_tools()}}
    elif method == "tools/call":
        result = server["tools"].call(params["name"], params.get("arguments", {}))
        return {"result": result}
    elif method == "resources/list":
        return {"result": {"resources": server["resources"].list_resources()}}
    elif method == "resources/read":
        return {"result": server["resources"].read(params["uri"])}
    elif method == "prompts/list":
        return {"result": {"prompts": server["prompts"].list_prompts()}}
    elif method == "prompts/get":
        return {"result": server["prompts"].get(params["name"], params.get("arguments"))}
    return {"error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    server = build_mcp_server("my-mcp-server", "1.0.0")

    @server["tools"].tool("add", "Add two numbers", {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    })
    def add(a, b):
        return a + b

    req = handle_request(server, {"method": "tools/list"})
    print(f"Tools: {[t['name'] for t in req['result']['tools']]}")
    req2 = handle_request(server, {"method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}})
    print(f"add(2, 3) = {req2['result']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())