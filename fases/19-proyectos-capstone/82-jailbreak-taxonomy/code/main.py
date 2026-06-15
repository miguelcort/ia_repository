"""
Lección: 82-jailbreak-taxonomy
Fase: 19
Capstone de ingeniería AI: 82 Jailbreak Taxonomy.
"""
from __future__ import annotations
import sys

JAILBREAK_TAXONOMY = {
    "prompt_injection": {
        "direct": ["ignore previous", "system override"],
        "indirect": ["via tool output", "via RAG"],
        "defenses": ["tagging", "spotlighting", "STR"],
    },
    "jailbreak": {
        "roleplay": ["DAN", "evil twin"],
        "many_shot": ["256-shot dialogues"],
        "ascii": ["art-based"],
        "defenses": ["LlamaGuard", "constitutional"],
    },
}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
