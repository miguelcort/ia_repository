"""
Lección: 42-large-corpus-downloader
Fase: 19
Capstone de ingeniería AI: 42 Large Corpus Downloader.
"""
from __future__ import annotations
import sys

def download_fineweb(output_path, n_docs=100000):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True,
                     split="train")
    out = []
    for i, item in enumerate(ds):
        if i >= n_docs:
            break
        out.append({"text": item["text"], "id": item["id"]})
    return out



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
