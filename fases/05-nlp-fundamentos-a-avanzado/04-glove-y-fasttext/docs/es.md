# 04 — GloVe, FastText y sub-word embeddings

> Más allá de Word2Vec: GloVe usa factorización de matriz de co-ocurrencia global; FastText maneja sub-palabras para vocabularios raros y multi-idioma.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-embeddings-de-palabras-word2vec
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar el algoritmo de GloVe.
- Usar FastText para vocabularios raros y multi-idioma.
- Diagnosticar cuándo cada técnica es preferible.
- Conocer las limitaciones de los embeddings estáticos.

## El problema

Word2Vec (lección 03) usa ventanas de contexto local. No
aprovecha las estadísticas globales del corpus. GloVe
(Global Vectors) usa la **matriz de co-ocurrencia global** y
factoriza con SVD, capturando patrones más estables.
FastText (Bojanowski et al., 2017) representa cada palabra
como una bolsa de n-gramas de caracteres, manejando
palabras raras y vocabularios ricos en morfología.

## El concepto

**GloVe (Pennington et al., 2014).** Construye la matriz
de co-ocurrencia `X` donde `X_{ij}` es el número de veces
que la palabra `j` aparece en el contexto de la palabra
`i`. La **hipótesis** es que `log(X_{ij})` se descompone
como `w_i^T w_j + b_i + b_j`. La loss minimiza el
error cuadrático ponderado por `f(X_{ij})` (más peso a
co-ocurrencias frecuentes, pero con cap para evitar que
"the" domine).

**Ventajas GloVe vs Word2Vec.**

- **GloVe:** entrena sobre estadísticas globales, no solo
  ventanas locales. Más estable.
- **Word2Vec:** más simple, streaming, no necesita
  almacenar la matriz completa.

**FastText (Bojanowski et al., 2017).** Representa cada
palabra como una bolsa de n-gramas de caracteres. "where"
con n=3 se representa como `<wh`, `whe`, `her`, `ere`,
`re>`, más la palabra completa. La embedding de una
palabra es la suma de las embeddings de sus n-gramas.

**Ventajas FastText.**

- **OOV:** palabras nuevas (no en vocabulario) aún
  tienen n-gramas que sí están. Embedding razonable.
- **Morfología:** "correr", "corría", "corriendo"
  comparten n-gramas. Embeddings relacionados.
- **Idiomas morfológicamente ricos:** español, alemán,
  finlandés, turco. Word2Vec/GloVe pierden.

**Embeddings contextualizados.** Word2Vec, GloVe, y
FastText producen **embeddings estáticos**: una palabra
tiene UN vector sin importar el contexto. "banco"
(institución) y "banco" (asiento) tienen el mismo vector.
BERT y GPT producen embeddings **contextualizados** que
resuelven esto (lección 19, transformers).

**Pre-trained embeddings más usados en 2026.**

- **Word2Vec / GloVe / FastText:** clásicos. Para tareas
  con pocos datos y vocabularios controlados.
- **BERT / RoBERTa / DeBERTa:** contextualizados, SOTA
  en clasificación.
- **GPT embeddings / LLaMA embeddings:** de LLMs.
  Calidad SOTA pero más pesados.

**Trampas.**

- **Out-of-vocabulary en producción:** siempre usa
  FastText o sub-palabras para vocabularios abiertos.
- **Sesgo:** los embeddings heredan sesgos del corpus
  (género, raza). Auditar con fairness metrics.
- **Idioma no soportado:** los embeddings pre-entrenados
  son específicos del idioma. Para bajo-resource,
  entrena desde cero o usa mBERT.

## Constrúyelo

```python
import numpy as np


def glove_loss(w_i, w_j, b_i, b_j, x_ij):
    """Loss de GloVe para un par (i, j)."""
    weight = np.minimum(1.0, (x_ij / 100) ** 0.75) if x_ij < 100 else 1.0
    pred = w_i @ w_j + b_i + b_j
    return weight * (pred - np.log(max(x_ij, 1e-10))) ** 2


def fasttext_word_vector(word, n_grams, char_n_gram_vocab, word_vocab):
    """Embedding de FastText: suma de n-gramas de la palabra + embedding
    de la palabra completa si está en vocab."""
    emb = np.zeros_like(next(iter(word_vocab.values())))
    word_with_markers = f"<{word}>"
    for i in range(len(word_with_markers) - n_grams + 1):
        n_gram = word_with_markers[i:i + n_grams]
        if n_gram in char_n_gram_vocab:
            emb += char_n_gram_vocab[n_gram]
    if word in word_vocab:
        emb += word_vocab[word]
    return emb
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-embeddings-choice
fase: 05
leccion: 04
---

Eres un asistente que ayuda a elegir embeddings. Reci-
birás la tarea, el corpus, y los recursos. Tu trabajo:

1. Si corpus pequeño y vocabulario controlado: Word2Vec
   o GloVe pre-entrenados.
2. Si vocabulario abierto (chatbot, search): FastText.
3. Si idioma morfológicamente rico: FastText con
   subwords.
4. Si necesitas embeddings contextualizados: BERT o
   Sentence-BERT.
5. Para SOTA en classification: fine-tunea BERT.
6. Para búsqueda semántica: Sentence-BERT o embeddings
   de LLM.
7. Advertir contra usar embeddings estáticos cuando
   la polisemia importa (banco: institución vs asiento).
8. Recomienda siempre auditar sesgos en producción.
```

## Ejercicios

1. **GloVe**: implementa el algoritmo y compara con
   Word2Vec en analogías.
2. **FastText**: aplica a un corpus en español y
   compara con FastText pre-entrenado.
3. **Desafío**: entrena embeddings para un idioma de
   bajos recursos (quechua, aimara).

## Lecturas recomendadas

- *GloVe: Global Vectors for Word Representation* —
  Pennington et al., 2014.
- *Enriching Word Vectors with Subword Information
  (FastText)* — Bojanowski et al., 2017.
- fastText: <https://fasttext.cc>.
- gensim: <https://radimrehurek.com/gensim>.

---

> 📚 **Adaptación al español** de la lección "[GloVe, FastText and Sub-word Embeddings]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
