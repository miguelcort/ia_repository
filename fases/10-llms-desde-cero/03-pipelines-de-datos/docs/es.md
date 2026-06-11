# Pipelines de datos

> Pipeline para LLM pre-training: download (Common Crawl, C4, RedPajama, FineWeb, Dolma) → filter (quality, language, PII, NSFW) → dedup (MinHash, LSH) → tokenize → pack. Datasets SOTA: Common Crawl, C4 (750GB), RedPajama (1.2T), FineWeb (15T), Dolma (3T), The Pile (800GB). Dedup: MinHash+LSH (Datatrove, SLIM). Mixing: temperature, DoReMi (Google), Skill-it. Hoy: FineWeb + FineWeb-Edu + RedPajama-v2 son SOTA.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/01-tokenizers
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar filtros de calidad.
- Implementar language ID mock.
- Implementar dedup con hashing.
- Implementar PII filter.
- Diagnosticar data mixing.

## Constrúyelo

```python
def dedup_hashes(docs, prefix_len=64):
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc[:prefix_len].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-data-pipeline
fase: 10
leccion: 03
---

1. Download -> filter -> dedup -> tokenize.
2. Quality, lang, PII, NSFW.
3. MinHash + LSH dedup.
4. FineWeb, RedPajama, Dolma.
5. DoReMi, skill-it mixing.
```

## Ejercicios

1. **Pipeline**: implementar pipeline
   end-to-end en Common Crawl sample.
2. **Dedup**: implementar MinHash
   + LSH desde cero.
3. **Desafio**: data mixing con
   DoReMi-style optimization.

## Lecturas recomendadas

- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020) - C4
- "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" (Gao et al., 2020)
- "The RefinedWeb Dataset for Falcon LLM" (Penedo et al., 2023)
- "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (Xie et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Data Pipelines]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).