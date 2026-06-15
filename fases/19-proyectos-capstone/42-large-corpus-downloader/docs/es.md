# 42 — Large corpus downloader

> Corpus downloader: download, dedup, filter, tokenize para pre-training. Fuentes: FineWeb, RedPajama-v2, Dolma, The Pile, C4. Tools: datatrove, img2dataset, webdataset.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/30
**Tiempo estimado:** ~30 minutos

## Objetivos

- Download FineWeb.
- Dedup (MinHash, suffix array).
- Quality filter.
- Tokenize y store.

## Constrúyelo

```python
from datasets import load_dataset
import hashlib


def download_fineweb(output_dir, n_docs=100000):
    """Download FineWeb-Edu (educational web)."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
                     streaming=True, split="train")
    out = []
    for i, item in enumerate(ds):
        if i >= n_docs:
            break
        out.append({"text": item["text"],
                   "id": item["id"]})
    return out


def minhash_dedup(items, threshold=0.8, num_perm=128):
    """MinHash deduplication."""
    from datasketch import MinHash
    sigs = []
    for it in items:
        m = MinHash(num_perm=num_perm)
        for word in it["text"].split():
            m.update(word.encode())
        sigs.append(m)
    # ... compare pairs
    return items
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-corpus-download
fase: 19
leccion: 42
---

1. Source selection.
2. Download streaming.
3. Dedup (MinHash).
4. Quality filter.
5. Tokenize + store.
```

## Ejercicios

1. **FineWeb**: 100K docs
   download.
2. **Dedup**: 0.8 threshold.
3. **Desafío**: 1M docs,
   quality filter.

## Lecturas recomendadas

- "FineWeb" (Hugging Face 2024)
- "RedPajama-v2" (Together 2023)
- "datatrove" (Hugging Face 2024)
- "DataComp-LM" (Li 2024)

---

> 📚 **Adaptación al español** de la lección
> "[42-large-corpus-downloader]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
