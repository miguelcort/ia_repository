# 31 — Tokenized dataset con sliding window

> Sliding window: chunking de secuencias largas en windows de N tokens con stride S. Para pre-training: el modelo ve overlaps, aprende dependencies de largo alcance. Implementación: numpy memmap, HDF5.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/30, Fase 10/04
**Tiempo estimado:** ~25 minutos

## Objetivos

- Implementar sliding window.
- Stride vs no-overlap.
- Memmap / HDF5 storage.
- Benchmark throughput.

## Constrúyelo

```python
import numpy as np


def sliding_window_chunks(token_ids, window=2048, stride=1024):
    """Yield chunks de tokens con overlap."""
    for i in range(0, len(token_ids) - window, stride):
        yield token_ids[i:i + window]


def build_memmap_dataset(token_files, output_path,
                       window=2048, stride=1024):
    """Build np memmap de chunks."""
    all_tokens = []
    for f in token_files:
        all_tokens.extend(np.load(f, mmap_mode="r").tolist())
    chunks = list(sliding_window_chunks(all_tokens, window, stride))
    arr = np.array(chunks, dtype=np.int32)
    np.save(output_path, arr)
    return arr.shape
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sliding-window
fase: 19
leccion: 31
---

1. Sliding window.
2. Stride.
3. Memmap / HDF5.
4. Throughput benchmark.
5. Long-context.
```

## Ejercicios

1. **Window**: 2048 vs 4096
   vs 8192.
2. **Stride**: 512 vs 1024.
3. **Desafío**: 100GB
   corpus en memmap.

## Storage formats

Memmap (numpy): (1) Faster loading vs PyTorch DataLoader
para datasets grandes. (2) Single file, no fragmentation.
(3) dtype int32 (max vocab 2B). Limitación: file size
single, no sharding.

HDF5 (h5py): (1) Hierarchical, soporta groups/datasets.
(2) Compression (gzip, lz4). (3) Chunked storage. (4)
Memory-mapped. Standard para multi-modal datasets.

Zarr: (1) Cloud-native (S3, GCS). (2) Parallel writes.
(3) Compatible con numpy. (4) Webdataset integration.

Webdataset: (1) Tar-based, streaming. (2) Sharded.
(3) Multi-modal (text+image+audio). (4) Scale to
petabytes.

Sliding window: window W (context size, e.g. 2048),
stride S (overlap). S = W: no overlap, each token
seen once. S = W/2: 50% overlap, each token seen
twice. S = 1: full overlap, +infinite storage. Trade-
off: data efficiency vs storage vs learning signal.
Llama 3 uses 8K context, 8K stride (no overlap).

## Lecturas recomendadas

- "nanoGPT" (Karpathy 2022)
- "HDF5" (2024)
- "FineWeb" (Hugging Face 2024)
- "Webdataset" (Aiden 2020)
- "Zarr" (Alistair 2024)

---

> 📚 **Adaptación al español** de la lección
> "[31-tokenized-dataset-sliding-window]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
