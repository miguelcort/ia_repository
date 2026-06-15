"""
Lección: 64-chunking-strategies-advanced
Fase: 19
Capstone de ingeniería AI: 64 Chunking Strategies Advanced.
"""
from __future__ import annotations
import sys

def semantic_chunk(text, model, threshold=0.7):
    import numpy as np
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = (embeddings[i] @ embeddings[i - 1]) / (
            np.linalg.norm(embeddings[i])
            * np.linalg.norm(embeddings[i - 1]))
        if sim > threshold:
            chunks[-1].append(sentences[i])
        else:
            chunks.append([sentences[i]])
    return [". ".join(c) for c in chunks]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
