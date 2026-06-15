"""
Lección: 04-multimodal-document-qa
Fase: 19
Capstone de ingeniería AI: 04 Multimodal Document Qa.
"""
from __future__ import annotations
import sys

def colpali_retrieval(query, pdf_pages, model):
    """ColPali: multi-vector retrieval."""
    return model.encode(query) @ model.encode(pdf_pages).T


def multimodal_synthesis(query, pages, llm):
    """Synthesize answer with GPT-4V."""
    return llm(query, images=[p.to_image() for p in pages])



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
