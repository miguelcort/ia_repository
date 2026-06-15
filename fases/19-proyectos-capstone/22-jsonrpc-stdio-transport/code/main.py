"""
Lección: 22-jsonrpc-stdio-transport
Fase: 19
Capstone de ingeniería AI: 22 Jsonrpc Stdio Transport.
"""
from __future__ import annotations
import sys

import json
import sys


def jsonrpc_server(handlers):
    for line in sys.stdin:
        req = json.loads(line)
        method = req.get("method")
        req_id = req.get("id")
        if method in handlers:
            result = handlers[method](req.get("params", {}))
            if req_id is not None:
                print(json.dumps({"jsonrpc": "2.0", "id": req_id,
                                 "result": result}), flush=True)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
