"""
Lección: 17-mcp-gateways-and-registries
Fase: 13
MCP gateways: centralized routing de multiples MCP servers.
Registries: discovery de MCP servers. Load balancing, auth,
monitoring. Composable architectures.
"""
from __future__ import annotations
import time


class MCPRegistry:
    """Mock MCP server registry."""
    def __init__(self):
        self.servers = {}

    def register(self, name, url, version, capabilities, auth=None):
        """Register MCP server."""
        self.servers[name] = {
            "name": name,
            "url": url,
            "version": version,
            "capabilities": capabilities,
            "auth": auth,
            "registered_at": time.time(),
        }

    def lookup(self, capability):
        """Find servers con given capability."""
        return [
            s for s in self.servers.values()
            if capability in s["capabilities"]
        ]

    def list_servers(self):
        return list(self.servers.values())

    def unregister(self, name):
        self.servers.pop(name, None)


class MCPGateway:
    """Mock MCP gateway (proxy)."""
    def __init__(self, registry):
        self.registry = registry
        self.metrics = {"requests": 0, "errors": 0}

    def route(self, capability, request_fn):
        """Route request a un server con given capability."""
        candidates = self.registry.lookup(capability)
        if not candidates:
            self.metrics["errors"] += 1
            return None
        # simple round-robin (mock)
        server = candidates[self.metrics["requests"] % len(candidates)]
        self.metrics["requests"] += 1
        return {"server": server["name"], "result": request_fn(server)}

    def forward(self, server_name, request):
        """Forward request a specific server."""
        server = self.registry.servers.get(server_name)
        if not server:
            return None
        return {"server": server_name, "result": f"forwarded: {request}"}


def main() -> int:
    reg = MCPRegistry()
    reg.register("weather", "stdio://weather", "1.0", ["tools", "resources"])
    reg.register("search", "http://search", "1.0", ["tools"])
    gw = MCPGateway(reg)
    # route a un server
    res = gw.route("tools", lambda s: f"called {s['name']}")
    print(f"Routed: {res}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())