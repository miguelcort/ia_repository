# 13 — Sistemas de preguntas y respuestas

> Question Answering (QA): dada una pregunta en lenguaje natural, encontrar la respuesta en un corpus, base de datos, o documento.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-traduccion-automatica,
                  14-recuperacion-de-informacion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar extractive QA con BERT.
- Aplicar RAG (Retrieval-Augmented Generation) para
  open-domain QA.
- Diagnosticar métricas: EM, F1, Recall@K.
- Conocer variantes: multi-hop, long-form, multimodal QA.

## El problema

QA puede ser **closed-domain** (respuestas en un
documento dado, como SQuAD) o **open-domain** (respuestas
sobre todo el conocimiento, como en chatbots). La
arquitectura moderna dominante es RAG: recuperar
documentos relevantes con embeddings y luego usar un LLM
para generar la respuesta. La lección cubre los métodos
clásicos y modernos.

## El concepto

**SQuAD (Stanford Question Answering Dataset).** El
benchmark canónico de extractive QA: pregunta + pasaje,
el modelo predice el span de inicio y fin de la respuesta.
BERT resuelve SQuAD con una cabeza de QA (dos clasificadores
sobre cada token, start y end).

**Métricas de QA extractive.**

- **EM (Exact Match):** el span predicho coincide
  exactamente con el ground truth.
- **F1:** media armónica de precision y recall a nivel
  de tokens.

**Open-domain QA.** El modelo tiene acceso a una
colección de documentos (e.g. Wikipedia). Necesita:
1. **Retriever:** encontrar documentos relevantes.
2. **Reader:** extraer la respuesta del documento.

**DPR (Dense Passage Retrieval, Karpukhin et al.,
2020).** Retriever basado en BERT: dos encoders
independientes (pregunta, pasaje) entrenados con
contrastive loss. Más rápido y preciso que BM25.

**RAG (Lewis et al., 2020).** Combina un retriever
(DPR o BM25) con un generador seq2seq (BART). El
generador produce la respuesta condicionada en los
documentos recuperados.

**RAG moderno (2024+).** Retriever denso (e5, BGE,
Contriever) + LLM (GPT-4, Claude, LLaMA). El LLM lee
los top-K documentos y genera la respuesta con citas.
Más simple y flexible que el RAG clásico.

**Long-form QA.** Generar respuestas largas y
detalladas (vs un span). Requiere LLMs. Métricas:
long-form eval, factuality.

**Multi-hop QA.** Responder preguntas que requieren
combinar información de múltiples documentos. HotpotQA
es el benchmark canónico.

**Multimodal QA.** Combinar texto e imagen. La pregunta
puede referirse a un detalle visual. Modelos: Qwen-VL,
GPT-4V, LLaVA.

**Trampas.**

- **Sin grounding:** el LLM alucina en lugar de citar los
  documentos. Usar prompting que exija citas.
- **Retriever malo:** si el retriever no encuentra los
  documentos relevantes, el LLM no puede responder.
  Evaluar Recall@K del retriever.
- **Documentos duplicados:** el chunking puede generar
  duplicados. Usar deduplicación o MMR.

## Constrúyelo

```python
import numpy as np


class SimpleRetriever:
    """Retriever TF-IDF simplificado."""

    def __init__(self, documents):
        from collections import Counter
        self.documents = documents
        # Vocabulario
        self.vocab = {}
        for doc in documents:
            for word in doc.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        # TF-IDF simplificado
        n = len(documents)
        self.doc_tfidf = []
        for doc in documents:
            tf = Counter(doc.lower().split())
            total = sum(tf.values())
            vec = np.zeros(len(self.vocab))
            for w, c in tf.items():
                if w in self.vocab:
                    # Sub-linear TF * IDF
                    idf = np.log(n / (1 + sum(
                        1 for d in documents if w in d.lower()
                    )))
                    vec[self.vocab[w]] = (1 + np.log(c)) * idf
            norm = np.linalg.norm(vec) or 1
            self.doc_tfidf.append(vec / norm)

    def query(self, question, k=3):
        q_words = question.lower().split()
        q_vec = np.zeros(len(self.vocab))
        from collections import Counter
        for w, c in Counter(q_words).items():
            if w in self.vocab:
                idf = np.log(len(self.documents) / (1 + sum(
                    1 for d in self.documents if w in d.lower()
                )))
                q_vec[self.vocab[w]] = (1 + np.log(c)) * idf
        norm = np.linalg.norm(q_vec) or 1
        q_vec = q_vec / norm
        scores = np.array(self.doc_tfidf) @ q_vec
        top_k = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_k]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-qa
fase: 05
leccion: 13
---

Eres un asistente que ayuda con question answering.
Recibirás la pregunta y los documentos. Tu trabajo:

1. Si tienes documentos: RAG con retriever denso
   (e5, BGE) + LLM.
2. Si tienes un solo documento: extractive QA
   (BERT fine-tuneado en SQuAD).
3. Si necesitas SOTA open-domain: GPT-4o o Claude
   con RAG.
4. Chunking: 256-512 tokens con overlap 50.
5. Retriever: e5-large o BGE-large.
6. Prompt: "Responde la pregunta basándote solo en el
   contexto. Cita la fuente con [n]."
7. Evaluar con EM, F1, Recall@K del retriever.
8. Advertir contra alucinaciones: verificar que la
   respuesta está en los documentos recuperados.
```

## Ejercicios

1. **Retriever TF-IDF**: implementa y aplica a un
   dataset de preguntas.
2. **Extractive QA**: usa BERT fine-tuneado en
   SQuAD.
3. **Desafío**: implementa RAG con un retriever
   denso y un LLM local.

## Lecturas recomendadas

- *SQuAD* — Rajpurkar et al., 2016.
- *DPR* — Karpukhin et al., 2020.
- *RAG* — Lewis et al., 2020.
- *Retrieval-Augmented Generation for Large Language
  Models: A Survey* — Gao et al., 2023.
- HuggingFace transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Question Answering Systems]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
