# 07 — POS tagging y parsing

> El POS (Part-of-Speech) tagging asigna categorías gramaticales (sustantivo, verbo, adjetivo). El parsing determina la estructura sintáctica. Son la base de muchas tareas de NLP.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-procesamiento-de-texto
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar POS tagging con HMM (Hidden Markov Model).
- Conocer los parsers: constituency y dependency.
- Usar spaCy para tagging y parsing en español.
- Diagnosticar aplicaciones downstream: NER, extracción
  de relaciones.

## El problema

Después de tokenizar, queremos entender la estructura
gramatical. POS tagging dice si "casa" es sustantivo o
verbo. Dependency parsing dice que "el gato negro" tiene
"gato" modificado por "el" (det) y "negro" (adj). Estos
análisis estructurales son la base de NER, extracción de
relaciones, y question answering.

## El concepto

**POS tags comunes.**

- **Sustantivo (NOUN):** casa, libro, Juan.
- **Verbo (VERB):** correr, ser, tener.
- **Adjetivo (ADJ):** grande, rojo.
- **Adverbio (ADV):** rápidamente, muy.
- **Determinante (DET):** el, la, un.
- **Preposición (ADP):** a, de, en, con.
- **Conjunción (CCONJ):** y, o, pero.
- **Pronombre (PRON):** yo, tú, él.

**HMM (Hidden Markov Model) para POS tagging.** Modelo
generativo: cada palabra tiene una distribución de
emisión sobre POS tags; cada POS tag tiene una
distribución de transición al siguiente. Viterbi
recupera la secuencia más probable de tags.

**Limitaciones de HMM.** Asume que el tag depende solo
del tag anterior (Markov de orden 1). Falla con
dependencias largas. Modelos modernos (CRF, BiLSTM,
BERT) lo superan.

**CRF (Conditional Random Field).** Discriminativo:
modela la probabilidad de la secuencia de tags
condicional a la secuencia de palabras. Mejor que HMM
en la mayoría de los casos. Base de muchos taggers de
producción hasta BERT.

**BiLSTM-CRF.** Encoder BiLSTM produce features
contextuales; CRF encima decodifica la mejor secuencia.
SOTA en muchos idiomas antes de BERT.

**BERT para POS tagging.** Fine-tunea BERT con un
clasificador por token. SOTA actual en la mayoría de
idiomas. spaCy v3+ usa transformers.

**Dependency parsing.** Asigna relaciones
sujeto-objeto-modificador entre palabras. Universal
Dependencies es el estándar跨lingüe.

- nsubj: nominal subject
- dobj: direct object
- amod: adjectival modifier
- advmod: adverbial modifier
- prep: prepositional modifier

**Constituency parsing.** Produce el árbol sintáctico
formal (NP, VP, PP). Más usado en investigación
lingüística que en producción.

**Aplicaciones downstream.**

- **NER:** usa POS tags para reducir el espacio de
  búsqueda ("Persona" suele ser sustantivo propio).
- **Extracción de relaciones:** el sujeto de un verbo
  es el argumento principal.
- **Question answering:** los sintagmas nominales son
  candidatos a respuestas.
- **Traducción automática:** preserva la estructura
  sintáctica entre idiomas.

**Trampas.**

- **Ambigüedad:** "banco" puede ser sustantivo o verbo
  según contexto. POS tagging no resuelve solo.
- **Idiomas no soportados:** spaCy cubre ~25 idiomas.
  Para bajo-resource, entrenar desde cero o usar
  Stanza.
- **Tagging como input único:** los parsers modernos
  usan contextual embeddings (BERT), no solo features
  superficiales.

## Constrúyelo

```python
import numpy as np
from collections import defaultdict


class HMMTagger:
    """HMM para POS tagging con Viterbi simplificado."""

    def __init__(self):
        self.tags = []
        self.start_prob = {}
        self.trans_prob = {}  # tag -> {next_tag: prob}
        self.emit_prob = {}   # tag -> {word: prob}

    def fit(self, sentences):
        """Entrena con pares (palabra, tag)."""
        tag_count = defaultdict(int)
        word_tag_count = defaultdict(lambda: defaultdict(int))
        tag_trans = defaultdict(lambda: defaultdict(int))
        for sent in sentences:
            prev = "<START>"
            for word, tag in sent:
                tag_count[tag] += 1
                word_tag_count[tag][word] += 1
                tag_trans[prev][tag] += 1
                prev = tag
            tag_trans[prev]["<END>"] += 1
        self.tags = list(tag_count.keys())
        # Probabilidades
        total = sum(tag_count.values())
        for tag, c in tag_count.items():
            self.start_prob[tag] = c / total
        for prev, trans in tag_trans.items():
            t = sum(trans.values())
            self.trans_prob[prev] = {k: v / t for k, v in trans.items()}
        for tag, words in word_tag_count.items():
            t = sum(words.values())
            self.emit_prob[tag] = dict(words)
            for w in self.emit_prob[tag]:
                self.emit_prob[tag][w] /= t

    def viterbi(self, sentence):
        """Decodifica la mejor secuencia de tags via Viterbi."""
        V = [{}]
        path = {}
        for tag in self.tags:
            V[0][tag] = (self.start_prob.get(tag, 1e-10)
                         * self.emit_prob.get(tag, {}).get(
                             sentence[0], 1e-10))
            path[tag] = [tag]
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}
            for tag in self.tags:
                options = {
                    prev: V[t - 1][prev] *
                    self.trans_prob.get(prev, {}).get(tag, 1e-10) *
                    self.emit_prob.get(tag, {}).get(sentence[t], 1e-10)
                    for prev in self.tags
                }
                best = max(options, key=options.get)
                V[t][tag] = options[best]
                new_path[tag] = path[best] + [tag]
            path = new_path
        # Tag final
        last = max(V[-1], key=V[-1].get)
        return path[last]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pos-parsing
fase: 05
leccion: 07
---

Eres un asistente que ayuda con POS tagging o parsing.
Recibirás el corpus y el idioma. Tu trabajo:

1. Si idioma es español/inglés/otros 20: spaCy con
   modelo pre-entrenado.
2. Para producción: spaCy v3+ con transformer
   (es_core_news_trf).
3. Para bajo-resource: entrenar BiLSTM-CRF o usar
   Stanza (Stanford).
4. Aplicaciones downstream: NER (entities), extracción
   de relaciones, QA.
5. Visualizar con displaCy.
6. Evaluar con UAS (Unlabeled Attachment Score) y
   LAS (Labeled Attachment Score) para dependency
   parsing.
```

## Ejercicios

1. **HMM**: implementa Viterbi y entrena en un
   corpus pequeño.
2. **spaCy**: aplica POS tagging y dependency parsing
   a un texto en español.
3. **Desafío**: entrena un CRF con sklearn-crfsuite
   para POS tagging en un corpus pequeño.

## Lecturas recomendadas

- *Speech and Language Processing* — Jurafsky & Martin.
- *Universal Dependencies*: <https://universaldependencies.org>.
- spaCy: <https://spacy.io>.
- Stanza: <https://stanfordnlp.github.io/stanza>.
- sklearn-crfsuite: <https://sklearn-crfsuite.readthedocs.io>.

---

> 📚 **Adaptación al español** de la lección "[POS Tagging and Parsing]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
