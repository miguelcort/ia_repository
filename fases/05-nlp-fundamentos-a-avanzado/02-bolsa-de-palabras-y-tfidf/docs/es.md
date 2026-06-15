# 02 — Bolsa de palabras y TF-IDF

> Las representaciones dispersas (bag-of-words, TF-IDF) son el baseline de NLP clásico. Simples, interpretables, y aún sorprendentemente efectivas en muchos casos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-procesamiento-de-texto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar Bag-of-Words (BoW) y TF-IDF desde cero.
- Diagnosticar cuándo BoW/TF-IDF supera a embeddings
  densos.
- Calcular n-gramas y aplicar a tareas de texto.
- Conocer las variantes: BM25, sub-linear TF, etc.

## El problema

Un modelo de ML no entiende strings. Necesitamos
convertir texto a un vector de números. La opción más
simple: contar palabras. Bag-of-Words representa cada
documento como un vector de frecuencias de palabras. TF-IDF
pondera las palabras por su rareza. Son representaciones
dispersas (la mayoría del vector es 0) y simples, pero
siguen siendo baselines fuertes para clasificación de
texto.

## El concepto

**Bag-of-Words (BoW).** Para cada documento, un vector
donde la entrada `i` es la frecuencia de la palabra `i`
en el vocabulario. Pierde el orden: "perro muerde hombre"
y "hombre muerde perro" tienen el mismo BoW.

**Vocabulario.** Se construye del corpus: las top-N
palabras más frecuentes, ignorando stopwords. Para
100k documentos, vocabulario de 10k-50k.

**TF-IDF (Term Frequency - Inverse Document Frequency).**
Pondera cada palabra por su rareza en el corpus:

```text
TF(t, d) = freq(t, d) / |d|  # frecuencia normalizada
IDF(t) = log(N / df(t))      # N docs, df docs con t
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

Palabras raras en el corpus (pero presentes en el doc)
tienen TF-IDF alto; palabras comunes (the, a, de) tienen
TF-IDF bajo. Mejora sobre BoW puro.

**N-gramas.** Incluir secuencias de N palabras: "New York"
como bigrama. Captura contexto local pero explota el
vocabulario.

**Normalización.** Antes de alimentar a un modelo:

- **L2 norm:** cada vector tiene norma 1 (cosine
  similarity).
- **Sub-linear TF:** `1 + log(freq)` en vez de freq.
  Reduce el impacto de palabras muy frecuentes.
- **BM25:** función de ranking con saturación. Es la
  base de ElasticSearch y Lucene.

**Cuándo usar BoW/TF-IDF vs embeddings.**

| Caso | Recomendación |
|---|---|
| Pocos datos (< 10k docs) | TF-IDF |
| Muchos datos + GPU | Embeddings (BERT, etc.) |
| Interpretabilidad crítica | TF-IDF (sparse, explicable) |
| Latencia mínima (CPU) | TF-IDF + linear model |

**Trampas.**

- **Vocabulario desbalanceado:** un dominio con jargon
  técnico infla el vocabulario sin información.
- **Out-of-vocabulary:** palabras nuevas en test
  desaparecen. Solución: hashing trick o sub-palabras.
- **Bigramas sin filtrar:** bigramas raras inflan el
  vocabulario. Usar min_df o chi-cuadrado.

## Constrúyelo

```python
import math
import re
from collections import Counter, defaultdict


def construir_vocabulario(documentos, max_vocab=10000, min_freq=2):
    """Top max_vocab palabras con freq >= min_freq."""
    counter = Counter()
    for doc in documentos:
        counter.update(re.findall(r"\b\w+\b", doc.lower()))
    palabras = [w for w, c in counter.most_common(max_vocab)
                if c >= min_freq]
    return {w: i for i, w in enumerate(palabras)}


def bow_vector(documento, vocab):
    """Vector BoW para un documento."""
    v = [0] * len(vocab)
    for palabra in re.findall(r"\b\w+\b", documento.lower()):
        if palabra in vocab:
            v[vocab[palabra]] += 1
    return v


def tfidf_vectors(documentos, vocab):
    """Calcula TF-IDF para una lista de documentos."""
    n = len(documentos)
    # Document frequency
    df = defaultdict(int)
    for doc in documentos:
        for palabra in set(re.findall(r"\b\w+\b", doc.lower())):
            if palabra in vocab:
                df[palabra] += 1
    # TF-IDF
    vectors = []
    for doc in documentos:
        tf = Counter(re.findall(r"\b\w+\b", doc.lower()))
        total = sum(tf.values())
        v = [0] * len(vocab)
        for palabra, freq in tf.items():
            if palabra in vocab:
                # Sub-linear TF
                tf_norm = 1 + math.log(freq) if freq > 0 else 0
                idf = math.log(n / (df[palabra] + 1)) + 1
                v[vocab[palabra]] = tf_norm * idf
        # L2 normalize
        norm = math.sqrt(sum(x ** 2 for x in v)) or 1
        v = [x / norm for x in v]
        vectors.append(v)
    return vectors
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

Eres un asistente que ayuda a decidir entre BoW/TF-IDF y
embeddings. Recibirás el corpus y los recursos. Tu trabajo:

1. Si corpus < 10k docs: TF-IDF con linear model.
2. Si corpus > 100k: considerar embeddings (BERT).
3. Si necesitas interpretabilidad: TF-IDF (sparse).
4. Si latencia en CPU es crítica: TF-IDF.
5. Min_df=2, max_df=0.95, max_features=10k-50k.
6. Sub-linear TF: 1 + log(freq).
7. L2 normalize siempre.
8. Para clasificación: SVM o LogisticRegression
   con TF-IDF es el baseline fuerte.
```

## Ejercicios

1. **BoW**: implementa y aplica a un dataset de spam.
2. **TF-IDF**: compara accuracy con BoW en clasificación
   de noticias.
3. **Desafío**: implementa BM25 y compara con TF-IDF
   en ranking de documentos.

## Lecturas recomendadas

- *Introduction to Information Retrieval* — Manning,
  Raghavan, Schütze (libre en línea).
- scikit-learn TfidfVectorizer: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>.
- BM25: <https://en.wikipedia.org/wiki/Okapi_BM25>.

---

> 📚 **Adaptación al español** de la lección "[Bag of Words and TF-IDF]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
