# 15 — Modelado de temas: LDA, BERTopic

> Topic modeling descubre los temas principales en una colección de documentos. LDA es el método clásico probabilístico; BERTopic usa embeddings modernos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-bolsa-de-palabras-y-tfidf
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar LDA (Latent Dirichlet Allocation) desde cero.
- Aplicar BERTopic con embeddings.
- Diagnosticar cuándo cada método es preferible.
- Visualizar topics con pyLDAvis o BERTopic plots.

## El problema

Tienes 100k artículos de noticias y quieres saber de qué
hablan. Topic modeling descubre los temas principales
(política, deportes, tecnología) automáticamente. LDA
(2003) es el método clásico probabilístico. BERTopic
(2020) usa embeddings modernos + UMAP + HDBSCAN para
encontrar clusters semánticos.

## El concepto

**LDA (Latent Dirichlet Allocation, Blei et al., 2003).**
Modelo generativo: cada documento es una mezcla de temas;
cada tema es una distribución sobre palabras. Entrena
con Gibbs sampling o variational inference. Asume que el
orden de las palabras no importa (bag-of-words).

**Limitaciones de LDA.**

- Asume bag-of-words (ignora orden).
- No aprovecha embeddings semánticos.
- Número de temas K debe ser especificado.
- Funciona mal en documentos cortos (tweets, headlines).

**BERTopic (Grootendorst, 2020).** Pipeline moderno:

1. **Embeddings:** codifica documentos con un modelo
   (Sentence-BERT, e5).
2. **Dimensionality reduction:** UMAP a 5-10 dim.
3. **Clustering:** HDBSCAN (encuentra clusters de
   densidad variable).
4. **Token representation:** c-TF-IDF (TF-IDF con
   pesos por cluster) para describir cada topic.
5. **Fine-tuning:** opcional, reduce outliers.

**Ventajas de BERTopic sobre LDA.**

- Captura semántica (sinónimos, paráfrasis).
- Encuentra el número de topics automáticamente.
- Mejor en documentos cortos.
- Visualización interactiva con plotly.

**Cuándo usar cada método.**

| Caso | Recomendación |
|---|---|
| Corpus pequeño (< 10k docs) | LDA o BERTopic |
| Corpus grande | BERTopic con embeddings |
| Documentos cortos (tweets) | BERTopic |
| Interpretabilidad rápida | LDA + pyLDAvis |
| Documentos largos, contexto | BERTopic con chunking |

**Métricas de evaluación.**

- **Coherence:** mide la coherencia semántica de los
  topics (palabras relacionadas entre sí).
- **Diversity:** qué tan distintos son los topics entre
  sí.
- **Perplexity (LDA):** qué tan bien predice el modelo
  una muestra held-out. Más bajo mejor.

**Trampas.**

- **Número de topics mal elegido:** muy pocos = temas
  demasiado generales; muy muchos = sobreajuste. Usar
  coherencia.
- **Stopwords en topics:** quitar stopwords antes de
  LDA/BERTopic.
- **Topics ruidosos:** algunos clusters son outliers
  sin tema claro. BERTopic los marca como tales.

## Constrúyelo

```python
import numpy as np
from collections import Counter


def lda_gibbs(docs, vocab, n_topics=5, n_iter=100, alpha=0.1, beta=0.1):
    """LDA simplificado con collapsed Gibbs sampling."""
    n_docs = len(docs)
    n_vocab = len(vocab)
    # Inicialización aleatoria
    z = []  # topic por palabra por documento
    for doc in docs:
        z.append([np.random.randint(n_topics) for _ in doc])
    # Conteos
    n_dk = np.zeros((n_docs, n_topics))  # doc-topic
    n_kw = np.zeros((n_topics, n_vocab))  # topic-word
    n_k = np.zeros(n_topics)
    for d, doc in enumerate(docs):
        for i, w in enumerate(doc):
            k = z[d][i]
            n_dk[d, k] += 1
            n_kw[k, w] += 1
            n_k[k] += 1
    # Gibbs sampling
    for it in range(n_iter):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k = z[d][i]
                n_dk[d, k] -= 1
                n_kw[k, w] -= 1
                n_k[k] -= 1
                # Distribución condicional
                p = (n_dk[d] + alpha) * (n_kw[:, w] + beta) / (
                    n_k + n_vocab * beta
                )
                p = p / p.sum()
                k_new = np.random.choice(n_topics, p=p)
                z[d][i] = k_new
                n_dk[d, k_new] += 1
                n_kw[k_new, w] += 1
                n_k[k_new] += 1
    return n_kw, n_dk, z


def topic_words(n_kw, vocab, n_words=10):
    """Devuelve las top-N palabras por topic."""
    topics = []
    for k in range(n_kw.shape[0]):
        top_idx = np.argsort(-n_kw[k])[:n_words]
        topics.append([list(vocab.keys())[list(vocab.values()).index(i)]
                       for i in top_idx])
    return topics
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-topic-modeling
fase: 05
leccion: 15
---

Eres un asistente que ayuda con topic modeling. Recibirás
el corpus y el caso de uso. Tu trabajo:

1. Si corpus pequeño: LDA con gensim o sklearn.
2. Si corpus grande o documentos cortos: BERTopic.
3. Embeddings: Sentence-BERT, e5, o BGE.
4. Reducción de dimensionalidad: UMAP.
5. Clustering: HDBSCAN (encuentra número de topics
   automáticamente).
6. Token representation: c-TF-IDF.
7. Evaluar con coherencia.
8. Visualizar con pyLDAvis (LDA) o BERTopic plotly.
9. Filtrar stopwords y palabras muy frecuentes/raras
   antes de entrenar.
```

## Ejercicios

1. **LDA**: implementa y entrena en un corpus de
   noticias.
2. **BERTopic**: aplica a un dataset de tweets.
3. **Desafío**: compara LDA vs BERTopic en coherencia
   y tiempo de entrenamiento.

## Lecturas recomendadas

- *Latent Dirichlet Allocation* — Blei et al., 2003.
- *BERTopic: Neural topic modeling with class-based
  TF-IDF* — Grootendorst, 2022.
- gensim: <https://radimrehurek.com/gensim>.
- BERTopic: <https://maartengr.github.io/BERTopic>.

---

> 📚 **Adaptación al español** de la lección "[Topic Modeling: LDA, BERTopic]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
