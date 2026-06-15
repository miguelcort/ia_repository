"""
Lección: 31-tokenized-dataset-sliding-window
Fase: 19
Capstone de ingeniería AI: 31 Tokenized Dataset Sliding Window.
"""
from __future__ import annotations
import sys

import numpy as np


def sliding_window_chunks(token_ids, window=2048, stride=1024):
    for i in range(0, len(token_ids) - window, stride):
        yield token_ids[i:i + window]


def build_memmap_dataset(token_files, output_path, window=2048,
                       stride=1024):
    all_tokens = []
    for f in token_files:
        all_tokens.extend(np.load(f, mmap_mode="r").tolist())
    chunks = list(sliding_window_chunks(all_tokens, window, stride))
    np.save(output_path, np.array(chunks, dtype=np.int32))
    return len(chunks)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
