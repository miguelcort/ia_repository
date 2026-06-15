"""
Lección: 83-prompt-injection-detector
Fase: 19
Capstone de ingeniería AI: 83 Prompt Injection Detector.
"""
from __future__ import annotations
import sys

def regex_injection_check(text):
    import re
    patterns = [r"ignore previous", r"system override",
               r"new instructions", r"forget everything"]
    return any(re.search(p, text.lower()) for p in patterns)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
