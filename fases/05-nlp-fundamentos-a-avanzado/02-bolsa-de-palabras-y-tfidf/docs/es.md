# Bolsa de palabras y TF-IDF

> La representacion de texto mas simple que funciona: vector de counts o TF-IDF. Sigue siendo el default para search engines, clasificacion con pocos datos, y como baseline antes de embeddings.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-procesamiento-de-texto
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Construir vocabulario y vectorizar.
- Calcular TF-IDF.
- Comparar documentos con similitud coseno.
- Diagnosticar limitaciones vs embeddings.

## Constrúyelo

```python
def tf_idf(docs, vocab):
    N = len(docs)
    tf = np.zeros((N, len(vocab)))
    for i, doc in enumerate(docs):
        for t in tokenizar(doc):
            if t in vocab:
                tf[i, vocab[t]] += 1
        tf[i] /= len(tokenizar(doc))
    df = np.array([sum(1 for d in docs if w in tokenizar(d)) for w in vocab])
    idf = np.log(N / (1.0 + df)) + 1.0
    return tf * idf
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bow-tfidf
fase: 05
leccion: 02
---

1. Clasif: TF-IDF + LR/SVM.
2. Search: TF-IDF + cosine + Faiss.
3. Topic: gensim LDA.
4. Spam: TF-IDF + LR threshold.
5. max_df=0.95, min_df=2, ngrams (1,2).
```

## Ejercicios

1. **N-gramas**: extender TF-IDF a bigramas y trigramas.
2. **Sublinear TF**: implementar log normalization.
3. **Desafio**: clasificador de spam en SMS con TF-IDF +
   LR, alcanzar F1 > 0.95 en dataset publico.

## Lecturas recomendadas

- "Introduction to Information Retrieval" (Manning et al., 2008)
- scikit-learn TfidfVectorizer: <https://scikit-learn.org/>
- gensim: <https://radimrehurek.com/gensim/>

---

> 📚 **Adaptación al español** de la lección "[Bag of Words and TF-IDF]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).