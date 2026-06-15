"""
Lección: 43-hdf5-tokenized-corpus
Fase: 19
Capstone de ingeniería AI: 43 Hdf5 Tokenized Corpus.
"""
from __future__ import annotations
import sys

import h5py
import numpy as np


def build_hdf5_corpus(token_batches, output_path,
                    chunk_size=100000):
    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("tokens", shape=(0,), maxshape=(None,),
                               chunks=(chunk_size,), dtype="int32",
                               compression="gzip")
        offset = 0
        for batch in token_batches:
            arr = np.array(batch, dtype=np.int32)
            dset.resize(offset + len(arr), axis=0)
            dset[offset:offset + len(arr)] = arr
            offset += len(arr)
    return offset



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
