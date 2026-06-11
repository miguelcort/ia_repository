# Tokenización de subpalabras

> La solucion al OOV: dividir palabras en subwords para un vocabulario fijo. BPE (GPT, RoBERTa), WordPiece (BERT), Unigram (T5, mBART), SentencePiece (framework). Maneja cualquier palabra, vocabulario compacto, multilingue.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18-nlp-multilingue
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entrenar BPE con greedy merges.
- Aplicar merges a texto.
- Implementar WordPiece longest-prefix.
- Diagnosticar BPE vs WordPiece vs Unigram.

## Constrúyelo

```python
def bpe_train(corpus, num_merges=10):
    # Iterativamente merge del par mas frecuente
    for _ in range(num_merges):
        pares = defaultdict(int)
        for palabra, split in splits.items():
            for i in range(len(split) - 1):
                pares[(split[i], split[i+1])] += vocab[palabra]
        mejor = max(pares, key=pares.get)
        # Aplicar merge
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
fase: 05
leccion: 19
---

1. Default: tokenizer del modelo preentrenado.
2. LLM: GPT/LLaMA BPE/SentencePiece.
3. Custom: SentencePiece sobre 1-10GB corpus.
4. Multilingual: mBERT, mT5, XLM-R.
5. 30-50K LLM, 100-250K multilingual.
```

## Ejercicios

1. **BPE completo**: implementar BPE training y apply.
2. **WordPiece**: implementar wordpiece_tokenizer con
   likelihood scoring.
3. **Desafio**: entrenar SentencePiece Unigram sobre
   un corpus espanol (1GB) y comparar coverage con
   tokenizer de BETO.

## Lecturas recomendadas

- "BPE" (Sennrich et al., 2016)
- "SentencePiece" (Kudo & Richardson, 2018)
- HuggingFace Tokenizers: <https://huggingface.co/docs/tokenizers/>

---

> 📚 **Adaptación al español** de la lección "[Subword Tokenization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).