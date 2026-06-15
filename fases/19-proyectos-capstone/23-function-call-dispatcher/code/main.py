"""
Lección: 23-function-call-dispatcher
Fase: 19
Capstone de ingeniería AI: 23 Function Call Dispatcher.
"""
from __future__ import annotations
import sys

def dispatch_tool_call(tool_call, tool_registry):
    name = tool_call["name"]
    args = tool_call["arguments"]
    tool = tool_registry.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}"}
    result = tool["fn"](**args)
    return {"result": str(result), "tool_call_id":
            tool_call.get("id")}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
