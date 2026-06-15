# 30 — BPE tokenizer from scratch

> BPE (Byte Pair Encoding, Sennrich 2016): greedy merge de pairs más frecuentes. Implementación: counts → merge → new vocab. BBPE (byte-level) para GPT/Llama. Rust implementation: tiktoken, Hugging Face tokenizers.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 10/01-tokenizers
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar BPE desde cero.
- BBPE (byte-level).
- Encoding/decoding.
- Eval speed vs tiktoken.

## Constrúyelo

```python
import collections


def get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def bpe_train(corpus, vocab_size):
    """Train BPE: counts -> merge -> vocab."""
    vocab = {tuple(word) + ("</w>",): count
            for word, count in corpus.items()}
    merges = []
    while len(vocab) < vocab_size:
        pairs = collections.Counter()
        for word, count in vocab.items():
            pairs.update(get_pairs(word))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_pair(best, vocab)
    return vocab, merges
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bpe
fase: 19
leccion: 30
---

1. Pair counting.
2. Greedy merge.
3. Vocab construction.
4. Encoding/decoding.
5. Compare tiktoken.
```

## Ejercicios

1. **BPE**: train en 10K
   sentences.
2. **BBPE**: byte-level.
3. **Desafío**: 1M tokens,
   compare speed.

## Detalles de implementación

Train BPE iterativo: (1) Vocab init = bytes (BBPE)
o chars. (2) Loop: cuenta adjacent pairs en corpus.
(3) Merge el pair más frecuente (nuevo token). (4)
Add token a vocab. (5) Repeat hasta vocab_size. BBPE
(GPT-2, GPT-3, Llama) usa bytes 0-255 como base, lo
que elimina OOV total. Word-level BPE (Sennrich
2016) parte de words. Unigram (Kudo 2018, T5) es
alternativa probabilística top-down.

Encoding: greedy longest-match. Decoding: token → bytes
→ string. Special tokens (BOS, EOS, PAD, UNK) se
agregan al vocab. Regex pre-tokenization separa words
(GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+|...`).

Optimización: pre-tokenize + cache, paralelizar con
Ray/multiprocessing, sorted vocab por frecuencia.
Tamaños típicos: 32K (BERT), 50K (GPT-2), 100K-200K
(Llama 3 multilingual). Train: 1-3 horas para 100B
tokens en 64 cores.

Rust implementations: tiktoken (OpenAI, 10x faster
than Python), tokenizers (Hugging Face, 100x). Python
puro es para aprender; producción usa Rust.

## Lecturas recomendadas

- "BPE" (Sennrich 2016)
- "tiktoken" (OpenAI)
- "tokenizers" (Hugging Face)
- "Unigram" (Kudo 2018)

---

> 📚 **Adaptación al español** de la lección
> "[30-bpe-tokenizer-from-scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
