"""
Lección: 13-mcp-server-with-registry
Fase: 19
Capstone de ingeniería AI: 13 Mcp Server With Registry.
"""
from __future__ import annotations
import sys

def mcp_server(tools):
    """MCP server with tools."""
    app = Server("my-server")
    app.list_tools = lambda: [Tool(name=t["name"],
                                  description=t["description"],
                                  inputSchema=t["schema"])
                             for t in tools]
    app.call_tool = lambda name, args: tools[name]["fn"](**args)
    return app



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
