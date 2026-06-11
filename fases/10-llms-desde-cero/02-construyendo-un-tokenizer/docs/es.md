# Construyendo un tokenizer

> BPE desde cero: pre-tokenize (GPT-2 regex, SentencePiece bytes), init vocab con chars, iterativamente merge del pair más frecuente hasta `vocab_size`. Encode: aplicar merges en orden, lookup id. Decode: id → token, concatenar. BBPE (byte-level) es estándar moderno: vocab inicial = 256 bytes UTF-8, sin OOV, multilingual. Frameworks: HF tokenizers (Rust), SentencePiece, tiktoken (OpenAI). Roundtrip lossless.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/01-tokenizers
**Tiempo estimado:** ~35 minutos

## Objetivos

- Implementar pre-tokenization GPT-2.
- Implementar BPE training (init + merge iterativo).
- Implementar encode/decode.
- Diagnosticar BBPE vs char-level.

## Constrúyelo

```python
class BPETokenizer:
    def train(self, corpus, vocab_size):
        word_freqs = Counter()
        for text in corpus:
            for token in pre_tokenize(text):
                word_freqs[token] += 1
        # Init vocab with chars
        chars = {ch for word in word_freqs for ch in word}
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        # Iteratively merge
        while len(self.vocab) < vocab_size:
            pairs = self._get_pair_freqs(word_syms, word_freqs)
            best = max(pairs, key=pairs.get)
            ...
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bpe-from-scratch
fase: 10
leccion: 02
---

1. Pre-tokenize.
2. BPE train.
3. Encode/decode.
4. BBPE byte-level.
5. HF, tiktoken, SentencePiece.
```

## Ejercicios

1. **BPE from scratch**: entrenar
   tokenizer en corpus de Shakespeare.
2. **BBPE**: implementar byte-level.
3. **Desafio**: tokenizer para
   codigo (CodeLlama-style).

## Lecturas recomendadas

- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)

---

> 📚 **Adaptación al español** de la lección "[Building a Tokenizer]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).