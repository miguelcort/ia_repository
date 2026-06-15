"""
Lección: 54-paper-writer
Fase: 19
Capstone de ingeniería AI: 54 Paper Writer.
"""
from __future__ import annotations
import sys

def write_paper_sections(results, related_work, llm):
    return {
        "abstract": llm(f"Write abstract: {results}"),
        "intro": llm(f"Write intro citing: {related_work}"),
        "method": llm(f"Write methods: {results['method']}"),
        "results": llm(f"Write results: {results['data']}"),
        "discussion": llm(f"Write discussion: {results['conclusions']}"),
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
