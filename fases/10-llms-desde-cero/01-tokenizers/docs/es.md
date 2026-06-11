# Tokenizers

> Tokenizers: convierten texto a secuencia de tokens. Tipos: char-level, word-level, subword (BPE, WordPiece, Unigram, SentencePiece). BPE: greedy merge por frequency (GPT, Llama). WordPiece: greedy likelihood (BERT). Unigram: probabilistic top-down (T5, ALBERT). BBPE (byte-level) es estándar moderno. Special tokens: BOS, EOS, PAD, MASK. Vocab sizes: 32K-200K. Frameworks: Hugging Face tokenizers, SentencePiece, tiktoken.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar char y word-level tokenization.
- Implementar BPE training y tokenization.
- Comparar BPE, WordPiece, Unigram.
- Diagnosticar special tokens.

## Constrúyelo

```python
def bpe_train_step(vocab, num_merges=10):
    merges = []
    for _ in range(num_merges):
        pairs = compute_pair_frequencies(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        # Merge
        new_vocab = {}
        for token, count in vocab.items():
            new_token = token.replace(f"{best[0]} {best[1]}", f"{best[0]}{best[1]}")
            new_vocab[new_token] = count
        vocab = new_vocab
        merges.append(best)
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
name: prompt-tokenizer
fase: 10
leccion: 01
---

1. BPE greedy merge, BBPE.
2. WordPiece, Unigram.
3. Vocab 32K-200K.
4. Special tokens.
5. tiktoken, HF, SentencePiece.
```

## Ejercicios

1. **BPE**: entrenar BPE en corpus
   pequeno, ver merges.
2. **Tokenizer**: comparar char vs
   word vs BPE en compresion.
3. **Desafio**: entrenar SentencePiece
   en multilingual corpus.

## Lecturas recomendadas

- "Byte Pair Encoding" (Sennrich et al., 2016)
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
- "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)
- "tiktoken" (OpenAI)

---

> 📚 **Adaptación al español** de la lección "[Tokenizers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).