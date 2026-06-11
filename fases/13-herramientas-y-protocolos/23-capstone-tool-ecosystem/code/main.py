"""
Lección: 23-capstone-tool-ecosystem
Fase: 13
Capstone: integracion de tools, MCP, A2A, routing, observability.
Build a complete tool ecosystem.
"""
from __future__ import annotations
import time


class ToolEcosystem:
    """Mock end-to-end tool ecosystem."""
    def __init__(self):
        self.tools = {}
        self.metrics = {"requests": 0, "errors": 0, "fallback_used": 0}
        self.audit_log = []

    def register_tool(self, name, handler, cost=1.0, capability="general"):
        """Register tool."""
        self.tools[name] = {
            "name": name,
            "handler": handler,
            "cost": cost,
            "capability": capability,
        }

    def route(self, capability):
        """Route to cheapest tool with given capability."""
        candidates = [t for t in self.tools.values() if t["capability"] == capability]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t["cost"])

    def call(self, tool_name, args, fallback_chain=None):
        """Call tool con fallback."""
        self.metrics["requests"] += 1
        if tool_name not in self.tools:
            self.metrics["errors"] += 1
            if fallback_chain:
                self.metrics["fallback_used"] += 1
                return self.call(fallback_chain[0], args, fallback_chain=fallback_chain[1:])
            return None
        try:
            result = self.tools[tool_name]["handler"](**args)
            self.audit_log.append({
                "timestamp": time.time(),
                "tool": tool_name,
                "args": args,
                "result": "ok",
            })
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            if fallback_chain:
                self.metrics["fallback_used"] += 1
                return self.call(fallback_chain[0], args, fallback_chain=fallback_chain[1:])
            self.audit_log.append({
                "timestamp": time.time(),
                "tool": tool_name,
                "args": args,
                "result": f"error: {e}",
            })
            return None

    def observe(self):
        """Return metrics."""
        return self.metrics


def main() -> int:
    eco = ToolEcosystem()
    eco.register_tool("primary", lambda x: f"primary({x})", cost=5.0, capability="search")
    eco.register_tool("fallback", lambda x: f"fallback({x})", cost=1.0, capability="search")
    eco.register_tool("alt", lambda x: f"alt({x})", cost=2.0, capability="search")
    # route
    tool = eco.route("search")
    print(f"Routed to: {tool['name']}")
    # call
    result = eco.call("primary", {"x": 1}, fallback_chain=["fallback", "alt"])
    print(f"Result: {result}")
    # metrics
    print(f"Metrics: {eco.observe()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())