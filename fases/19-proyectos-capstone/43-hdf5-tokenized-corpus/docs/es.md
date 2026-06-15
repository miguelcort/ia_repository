# 43 — HDF5 tokenized corpus

> HDF5 tokenized corpus: efficient storage for billions of tokens. Format: chunks of int32 token ids, byte-compressed. Libraries: h5py, zarr, webdataset. Streaming reads en training.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/31, 19/42
**Tiempo estimado:** ~25 minutos

## Objetivos

- HDF5 storage format.
- Chunks de tokens.
- Random access.
- Streaming reads.

## Constrúyelo

```python
import h5py
import numpy as np


def build_hdf5_corpus(token_batches, output_path,
                    chunk_size=100_000):
    """Build HDF5 con chunks de tokens."""
    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("tokens", shape=(0,),
                               maxshape=(None,),
                               chunks=(chunk_size,),
                               dtype="int32",
                               compression="gzip")
        offset = 0
        for batch in token_batches:
            arr = np.array(batch, dtype=np.int32)
            dset.resize(offset + len(arr), axis=0)
            dset[offset:offset + len(arr)] = arr
            offset += len(arr)
    return offset


def stream_load_hdf5(path, batch_size=2048, stride=1024):
    """Stream reads con sliding window."""
    with h5py.File(path, "r") as f:
        tokens = f["tokens"]
        for i in range(0, len(tokens) - batch_size, stride):
            yield tokens[i:i + batch_size]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-hdf5-corpus
fase: 19
leccion: 43
---

1. HDF5 dataset.
2. Chunks + compression.
3. Streaming reads.
4. Sliding window.
```

## Ejercicios

1. **Build**: 1M tokens
   HDF5.
2. **Stream**: training
   loop.
3. **Desafío**: 10B tokens
   corpus.

## Lecturas recomendadas

- "h5py" (2024)
- "zarr" (Alistair 2024)
- "FineWeb" (Hugging Face 2024)

---

> 📚 **Adaptación al español** de la lección
> "[43-hdf5-tokenized-corpus]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
