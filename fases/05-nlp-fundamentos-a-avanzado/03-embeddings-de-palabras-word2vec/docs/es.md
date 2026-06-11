# Embeddings de palabras con word2vec

> La idea que cambio NLP: las palabras son vectores. Palabras similares -> vectores cercanos. CBOW o Skip-gram con negative sampling. La base de los embeddings modernos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-bolsa-de-palabras-y-tfidf
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entender Skip-gram con negative sampling.
- Calcular similitud coseno entre embeddings.
- Resolver analogias (king - man + woman = queen).
- Diagnosticar word2vec vs BERT.

## Constrúyelo

```python
def skip_gram_step(target_emb, context_emb, vocab_size, n_neg=5, semilla=0):
    p_pos = sigmoid(context_emb @ target_emb)
    loss_pos = -np.log(p_pos + 1e-10)
    neg = rng.normal(0, 0.1, size=(n_neg, D))
    p_neg = sigmoid(neg @ target_emb)
    loss_neg = -np.log(1 - p_neg + 1e-10).sum()
    return loss_pos + loss_neg
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-embeddings
fase: 05
leccion: 03
---

1. Baseline: word2vec gensim, GloVe, fastText.
2. Multilingual: BPEmb, XLM-R.
3. Production: sentence-transformers.
4. Contexto: BERT, RoBERTa.
5. OOV: fastText subword.
```

## Ejercicios

1. **Entrenar word2vec**: usar gensim Word2Vec en
   un corpus espanol.
2. **Analogias**: implementar un evaluador de analogias
   (Google analogy test).
3. **Desafio**: entrenar word2vec en un corpus grande
   (e.g. 1B palabras) y visualizar con t-SNE.

## Lecturas recomendadas

- "Efficient Estimation of Word Representations in Vector
  Space" (Mikolov et al., 2013) — word2vec
- "Distributed Representations of Words and Phrases" (Mikolov
  et al., 2013) — negative sampling
- gensim: <https://radimrehurek.com/gensim/models/word2vec.html>

---

> 📚 **Adaptación al español** de la lección "[Word Embeddings with word2vec]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).