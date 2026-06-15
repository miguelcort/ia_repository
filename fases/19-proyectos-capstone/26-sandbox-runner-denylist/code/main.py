"""
Lección: 26-sandbox-runner-denylist
Fase: 19
Capstone de ingeniería AI: 26 Sandbox Runner Denylist.
"""
from __future__ import annotations
import sys

import re
import subprocess


DANGEROUS = [r"rm\s+-rf\s+/", r"curl.*\|.*bash",
            r"chmod\s+777", r":\(\)\{\s*:\|\:&\s*\}" ]


def sandboxed_exec(cmd, timeout=30):
    if any(re.search(p, cmd) for p in DANGEROUS):
        raise PermissionError(f"Dangerous: {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True,
                         timeout=timeout, text=True)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
