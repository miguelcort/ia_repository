# 08 — CNNs y RNNs para texto

> Antes de los transformers, las CNNs y RNNs eran las arquitecturas dominantes para texto. Aún son útiles: más rápidas y simples, especialmente para tareas cortas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-nucleo-deep-learning,
                  04-fundamentos-ml
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar TextCNN (convoluciones 1D para texto).
- Implementar LSTM y BiLSTM para clasificación de texto.
- Comparar CNN vs RNN vs Transformer para texto.
- Diagnosticar cuándo las arquitecturas clásicas siguen
  siendo útiles.

## El problema

Los transformers dominan NLP desde 2017, pero las CNNs y
RNNs siguen siendo útiles: más rápidas, menos memoria, y
suficientes para muchas tareas (clasificación de
sentimiento, NER). La lección cubre las arquitecturas
clásicas que aún verás en producción y en papers
recientes.

## El concepto

**TextCNN (Kim, 2014).** Convolución 1D sobre embeddings
de palabras. Cada filtro captura n-gramas de diferente
longitud (3, 4, 5 palabras). Max-pooling extrae el
feature más importante. Fully-connected al final.
Sorprendentemente efectivo en clasificación de texto
cortos.

**LSTM (Hochreiter & Schmidhuber, 1997).** RNN con
celda de memoria y tres puertas (input, forget, output).
Aprende dependencias largas mejor que vanilla RNN.
Resuelve el vanishing gradient parcialmente.

**GRU (Cho et al., 2014).** Variante simplificada del
LSTM con dos puertas (reset, update). Menos parámetros,
similar accuracy. Útil cuando el dataset es pequeño.

**BiLSTM.** Procesa la secuencia en ambas direcciones.
Captura contexto pasado y futuro. Especialmente útil en
NER y POS tagging.

**LSTM apilado (stacked).** Múltiples capas de LSTM.
Cada capa procesa la salida de la anterior. Permite
representaciones jerárquicas.

**Embeddings pre-entrenados.** En vez de aprender
embeddings desde cero, inicializa con Word2Vec, GloVe, o
FastText. Mejora la convergencia y la accuracy final en
datasets pequeños.

**Cuándo usar cada arquitectura.**

| Caso | Recomendación |
|---|---|
| Texto muy corto (< 100 palabras) | TextCNN |
| Texto largo con dependencias | BiLSTM |
| SOTA en accuracy | Transformer (BERT, etc.) |
| Latencia mínima en CPU | TextCNN |
| Few-shot | Transformer fine-tuneado |

**Trampas.**

- **Vanilla RNN:** vanishing gradient. Siempre usar
  LSTM o GRU.
- **Padding inadecuado:** batches con secuencias de
  longitud muy variable desperdician cómputo. Usar
  packed sequences o bucketing.
- **Embeddings congelados vs fine-tune:** congelar
  embeddings pre-entrenados es un regularizer. Fine-
  tunearlos es más expresivo pero puede sobreajustar
  con poco data.

## Constrúyelo

```python
import numpy as np


class TextCNN:
    """CNN 1D simplificada para clasificación de texto."""

    def __init__(self, vocab_size, embed_dim, n_classes,
                 filter_sizes=(2, 3, 4), n_filters=32):
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
        self.filters = {}
        for fs in filter_sizes:
            self.filters[fs] = np.random.randn(fs, embed_dim,
                                                n_filters) * 0.01
        self.W = np.random.randn(len(filter_sizes) * n_filters,
                                 n_classes) * 0.01

    def forward(self, token_ids):
        """token_ids: (batch, seq_len) -> logits (batch, n_classes)."""
        emb = self.embeddings[token_ids]  # (batch, seq, dim)
        pooled = []
        for fs, W in self.filters.items():
            # Convolución 1D: (batch, seq - fs + 1, n_filters)
            conv = np.zeros((emb.shape[0], emb.shape[1] - fs + 1, W.shape[2]))
            for i in range(emb.shape[1] - fs + 1):
                conv[:, i, :] = emb[:, i:i + fs, :] @ W.transpose(1, 0)
            pooled.append(conv.max(axis=1))
        h = np.concatenate(pooled, axis=1)  # (batch, sum_filters)
        return h @ self.W


class LSTMCell:
    """Celda LSTM simplificada."""

    def __init__(self, input_size, hidden_size):
        scale = np.sqrt(2 / (input_size + hidden_size))
        self.W_i = np.random.randn(input_size + hidden_size,
                                    4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h, c):
        concat = np.concatenate([x, h], axis=-1)
        gates = concat @ self.W_i + self.b
        i, f, g, o = np.split(gates, 4, axis=-1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = np.tanh(g)
        o = self.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-text-arch
fase: 05
leccion: 08
---

Eres un asistente que ayuda a elegir arquitectura para
texto. Recibirás el dataset, latencia objetivo, y
recursos. Tu trabajo:

1. Si quieres SOTA y tienes GPU: transformer
   fine-tuneado.
2. Si quieres velocidad y pocos datos: TextCNN.
3. Si dependencias largas (largo texto): BiLSTM.
4. Si latencia en CPU: TextCNN o DistilBERT.
5. Embeddings: inicializar con FastText pre-entrenado.
6. Para few-shot: usar embeddings de un LLM (e5, BGE)
   sin fine-tuning.
7. Recomienda monitorear gradientes para detectar
   vanishing/exploding en RNNs.
```

## Ejercicios

1. **TextCNN**: implementa y entrena en SST-2 o IMDB.
2. **BiLSTM**: implementa y compara accuracy con
   TextCNN.
3. **Desafío**: implementa un encoder-decoder LSTM
   para traducción.

## Lecturas recomendadas

- *Convolutional Neural Networks for Sentence
  Classification (TextCNN)* — Kim, 2014.
- *LSTM* — Hochreiter & Schmidhuber, 1997.
- *Speech and Language Processing* — Jurafsky & Martin.
- PyText: <https://github.com/facebookresearch/pytext>.

---

> 📚 **Adaptación al español** de la lección "[CNNs and RNNs for Text]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
