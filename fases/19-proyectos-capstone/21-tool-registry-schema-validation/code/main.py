"""
Lección: 21-tool-registry-schema-validation
Fase: 19
Capstone de ingeniería AI: 21 Tool Registry Schema Validation.
"""
from __future__ import annotations
import sys

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn, schema, permissions):
        self.tools[name] = {"fn": fn, "schema": schema,
                           "permissions": permissions}

    def execute(self, name, args, agent_id):
        tool = self.tools.get(name)
        if not tool or not self._check_permission(tool, agent_id):
            raise PermissionError(f"Denied: {name}")
        return tool["fn"](**args)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
