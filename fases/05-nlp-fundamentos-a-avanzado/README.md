# Fase 5 — NLP: de fundamentos a avanzado

> El lenguaje es la interfaz con la inteligencia.

Esta fase recorre el **procesamiento de lenguaje natural** desde los
métodos clásicos (Bolsa de Palabras, TF-IDF, embeddings estáticos)
hasta los modernos (RAG, evaluación de LLM, modelos de contexto
largo, grafos de conocimiento). La Fase 7 cubre los *transformers*
en detalle; esta fase los **usa** como bloques y se enfoca en las
tareas de NLP: tokenización, embeddings, clasificación, traducción,
resumen, NER, QA, chunking, RAG y evaluación.

El recorrido está organizado en **cuatro bloques**. El **bloque 1**
(lecciones 1–7) es la base: cómo se representa el texto y las tareas
clásicas de clasificación y extracción. El **bloque 2** (8–13) cubre
las redes neuronales aplicadas a texto: CNNs, RNNs, seq2seq,
atención, traducción, resumen, QA. El **bloque 3** (14–18) trata
tareas más maduras: recuperación, modelado de temas, generación
pre-Transformer, chatbots y multilingüismo. El **bloque 4** (19–29)
entra al ecosistema moderno: tokenización sub-palabra, RAG,
embeddings, *chunking*, *entity linking*, grafos de conocimiento y
evaluación de LLMs (RAGAS, DeepEval, NIAH, RULER).

## Índice de lecciones

### Bloque 1 — Representación y tareas básicas

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Procesamiento de texto](01-procesamiento-de-texto/) | Construir | Tokenización, normalización, stemming y lematización. |
| 02 | [Bolsa de palabras y TF-IDF](02-bolsa-de-palabras-y-tfidf/) | Construir | Representación dispersa, n-gramas y métricas. |
| 03 | [Embeddings Word2Vec desde cero](03-embeddings-de-palabras-word2vec/) | Construir | Skip-gram, CBOW y muestreo negativo. |
| 04 | [GloVe, FastText y sub-word embeddings](04-glove-y-fasttext/) | Construir | Co-ocurrencias, sub-palabras y OOV. |
| 05 | [Análisis de sentimiento](05-analisis-de-sentimiento/) | Construir | Lexicones, modelos clásicos y redes neuronales. |
| 06 | [Reconocimiento de entidades (NER)](06-reconocimiento-de-entidades/) | Construir | BIO tagging, CRF y modelos neuronales. |
| 07 | [POS tagging y parsing](07-etiquetado-pos-y-parsing/) | Construir | HMM, dependency parsing y constituency parsing. |

### Bloque 2 — Redes neuronales para texto

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 08 | [CNNs y RNNs para texto](08-cnns-y-rnns-para-texto/) | Construir | TextCNN, LSTM, GRU y *bidirectional*. |
| 09 | [Secuencia a secuencia](09-secuencia-a-secuencia/) | Construir | Encoder-decoder, *teacher forcing* y beam search. |
| 10 | [Mecanismo de atención](10-mecanismo-de-atencion/) | Construir | Atención aditiva, dot-product y scaled dot-product. |
| 11 | [Traducción automática](11-traduccion-automatica/) | Construir | NMT con attention y métricas BLEU. |
| 12 | [Resumen de texto](12-resumen-de-texto/) | Construir | Extractive vs abstractive, ROUGE. |
| 13 | [Sistemas de preguntas y respuestas](13-preguntas-y-respuestas/) | Construir | SQuAD, MRC y respuesta extractiva. |

### Bloque 3 — Tareas maduras y chatbots

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 14 | [Recuperación de información](14-recuperacion-de-informacion/) | Construir | BM25, índices invertidos, *dense retrieval*. |
| 15 | [Modelado de temas](15-modelado-de-temas/) | Construir | LDA, NMF y BERTopic. |
| 16 | [Generación de texto pre-Transformer](16-generacion-de-texto-pre-transformer/) | Construir | n-gramas, *sampling*, *temperature*. |
| 17 | [Chatbots: de reglas a redes neuronales](17-chatbots-de-reglas-a-neuronal/) | Construir | AIML, retrieval bots y *neural chatbots*. |
| 18 | [NLP multilingüe](18-nlp-multilingue/) | Construir | mBERT, XLM-R, *language ID*, *code-switching*. |

### Bloque 4 — Ecosistema moderno

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 19 | [Tokenización sub-word: BPE, WordPiece, Unigram](19-tokenizacion-de-subpalabras/) | Aprender | Algoritmos de tokenización y *SentencePiece*. |
| 20 | [Salidas estructuradas y decoding restringido](20-salidas-estructuradas-y-decoding-constrenido/) | Construir | JSON mode, *grammar-constrained decoding*, *logit processors*. |
| 21 | [Inferencia textual (NLI)](21-inferencia-de-texto-nli/) | Aprender | SNLI, MNLI, entailment y contradiction. |
| 22 | [Embeddings a fondo](22-modelos-de-embedding-a-profundidad/) | Aprender | Sentence-BERT, E5, BGE, *matryoshka*. |
| 23 | [Estrategias de chunking para RAG](23-estrategias-de-chunking-y-rag/) | Construir | Fixed, semantic, sliding window y *parent-document*. |
| 24 | [Resolución de correferencias](24-resolucion-de-coreferencias/) | Aprender | Pronombres, menciones y modelos neuronales. |
| 25 | [Entity linking y desambiguación](25-vinculacion-de-entidades/) | Construir | Wikification, BLINK, GENRE. |
| 26 | [Extracción de relaciones y grafos](26-extraccion-de-relaciones-y-grafo-de-conocimiento/) | Construir | REBEL, OpenIE, Knowledge Graphs. |
| 27 | [Evaluación de LLMs: RAGAS, DeepEval, G-Eval](27-frameworks-de-evaluacion-de-llm/) | Construir | Métricas de faithfulness, answer relevance, context. |
| 28 | [Evaluación de contexto largo: NIAH, RULER, LongBench, MRCR](28-evaluacion-de-contexto-largo/) | Aprender | Pruebas estándar de *long context*. |
| 29 | [Seguimiento de estado de diálogo](29-seguimiento-de-estado-de-dialogo/) | Construir | DST, slots, intents y *schema-guided*. |

## Prerrequisitos

- **Fases 0, 1, 2, 3 y 4** completas.
- Conocimiento de álgebra lineal y probabilidad.
- Familiaridad con PyTorch (Fase 3).
- Opcional: nociones básicas de *lingüística* (token, morfema,
  sintaxis) ayudan a las lecciones 1 y 7.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Tokenizar** texto en sub-palabras con BPE, WordPiece o
  Unigram y entender por qué los LLMs modernos no usan palabras
  completas.
- **Entrenar** un modelo de embeddings (Word2Vec o GloVe) y
  explorar la geometría resultante.
- **Construir** un sistema de clasificación de texto end-to-end
  (pre-procesamiento, embedding, modelo, evaluación).
- **Implementar** un sistema RAG con embeddings, retrieval,
  re-ranking y evaluación con RAGAS.
- **Evaluar** LLMs con métricas estándar (BLEU, ROUGE, BERTScore,
  faithfulness, answer relevance).
- **Diseñar** prompts y *decoders* con restricciones (JSON mode,
  *grammar-constrained decoding*) para producción.
- **Construir** un grafo de conocimiento a partir de texto con
  extracción de entidades y relaciones.

## Stack y herramientas

- **PyTorch** y **transformers** (Hugging Face).
- **tokenizers** y **sentencepiece** para tokenización.
- **spaCy** y **NLTK** para tareas clásicas.
- **scikit-learn** para TF-IDF, BM25 y LDA.
- **gensim** para Word2Vec y topic modeling.
- **rank_bm25, faiss, qdrant** para retrieval.
- **RAGAS, DeepEval, G-Eval** para evaluación.
- **BERTopic** para modelado de temas.
- **OpenAI/Anthropic/local** como LLMs en las últimas lecciones.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Tokenización** | Lección 01, 19 | Fase 10 (tokenizador de LLM) |
| **Embeddings** | Lecciones 3, 4, 22 | Fase 12 (multimodal), Fase 14 (RAG agent) |
| **Atención** | Lección 10 | Fase 7 (transformer) |
| **Seq2Seq** | Lección 09, 11 | Fase 7 (encoder-decoder) |
| **NER** | Lección 06 | Fase 14 (extracción para agentes) |
| **RAG** | Lecciones 14, 22, 23, 27 | Fase 14 (agentes RAG), Fase 19 (capstone) |
| **Knowledge Graph** | Lección 26 | Fase 14 (GraphRAG) |
| **Long context** | Lección 28 | Fase 10 (entrenamiento de LLM) |

## Cómo estudiar esta fase

1. **Bloque 1 es la base obligatoria.** Sin entender
   tokenización, TF-IDF y embeddings, las siguientes lecciones
   parecen magia.
2. **No intentes entrenar BERT desde cero.** Usa `transformers`
   y concéntrate en *fine-tunear* y *evaluar*.
3. **Las lecciones 19, 22, 23 y 27 son las más valiosas en
   2026** para ingenieros de LLM: tokenización, embeddings,
   chunking y evaluación.
4. **Construye un mini-proyecto integrador al final.** Una buena
   idea: un sistema RAG sobre tus propios documentos, evaluado
   con RAGAS.
5. **Las métricas importan más que el modelo.** Un LLM mediocre
   con buenas métricas de evaluación es preferible a un LLM
   excelente sin medición.

## Verificación de progreso

```bash
# Lección 03 — Word2Vec desde cero
python3 fases/05-nlp-fundamentos-a-avanzado/03-embeddings-de-palabras-word2vec/code/main.py

# Lección 19 — tokenizador BPE
python3 fases/05-nlp-fundamentos-a-avanzado/19-tokenizacion-de-subpalabras/code/main.py

# Lección 27 — RAGAS evaluation
python3 fases/05-nlp-fundamentos-a-avanzado/27-frameworks-de-evaluacion-de-llm/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Conexión con otras fases

- **Entrada** → [Fase 4 — Visión por computador](../04-vision-por-computador/README.md)
  y [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md).
- **Salida natural** → [Fase 7 — Transformers a fondo](../07-transformers-a-fondo/README.md)
  (la arquitectura de los transformers es la base de los LLMs).
- **Reuso en** → Fase 10 (LLM), Fase 11 (ingeniería de LLM),
  Fase 14 (agentes con RAG), Fase 19 (capstone).

## Recursos recomendados

- *Speech and Language Processing* — Jurafsky & Martin (PDF libre).
- *Foundations of Statistical Natural Language Processing* — Manning & Schütze.
- *Natural Language Processing with Transformers* — Tunstall, von Werra, Wolf.
- *Hugging Face NLP Course* — <https://huggingface.co/learn/nlp-course>.
- *spaCy course* — <https://course.spacy.io>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *tokenizer*,
  *embedding*, *attention*, *RAG*.
- [Fase 7 — Transformers a fondo](../07-transformers-a-fondo/README.md).
- [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
