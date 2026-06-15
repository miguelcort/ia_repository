"""
Lección: 51-literature-retrieval
Fase: 19
Capstone de ingeniería AI: 51 Literature Retrieval.
"""
from __future__ import annotations
import sys

def multi_source_search(query, n=20):
    from semanticscholar import SemanticScholar
    import arxiv
    arxiv_results = arxiv.Search(query, max_results=n)
    s2 = SemanticScholar()
    s2_results = s2.search_paper(query, limit=n)
    return {"arxiv": list(arxiv_results.results()),
            "s2": list(s2_results)}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
