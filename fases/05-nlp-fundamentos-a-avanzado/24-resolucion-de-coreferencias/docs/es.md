# 24 — Resolución de correferencias

> Coreference resolution: determinar cuándo dos menciones en un texto se refieren a la misma entidad ("Juan llegó. Él sonrió." → "Juan" y "Él" son la misma persona).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-etiquetado-pos-y-parsing
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar resolución de correferencias con reglas
  simples.
- Aplicar modelos neuronales (e.g. spaCy coref, LingMess).
- Diagnosticar métricas: MUC, B³, CEAF.
- Conectar a QA, resumen, y diálogo.

## El problema

En lenguaje natural, las entidades se mencionan de
múltiples formas: nombre propio ("Juan"), pronombre
("él"), descripción ("el científico"), alias ("Dr.
García"). La resolución de correferencias agrupa estas
menciones en clusters, cada uno representando una entidad.
Es esencial para QA, resumen coherente, y diálogo
multi-turno.

## El concepto

**Tipos de menciones.**

- **Pronombres:** él, ella, ello, ellos, este, ese.
- **Nombres propios:** Juan, Madrid, Apple.
- **Descripciones definidas:** "el gato", "la mesa".
- **Descripciones indefinidas:** "un gato", "una mesa".

**Tipos de correferencia.**

- **Identidad:** dos menciones se refieren al mismo
  referente. "Juan" y "Juan".
- **Anáfora pronominal:** un pronombre se refiere a un
  antecedente. "Juan llegó. Él sonrió."
- **Anáfora asociativa:** una mención activa un
  conocimiento del mundo. "Hojeamos el periódico. Las
  noticias eran alarmantes." (periódico → noticias).
- **Referenciabridging:** una mención introduce un
  nuevo referente relacionado. "Hojeamos el periódico.
  La primera página mostraba..."

**Algoritmos.**

- **Hobbs (1978):** algoritmo clásico basado en reglas
  de sintaxis. Recorre el árbol sintáctico de derecha a
  izquierda buscando el antecedente más cercano.
- **Rule-based (Haghighi & Klein, 2009):** sistema de
  reglas más elaborado. Base de Stanford CoreNLP.
- **Neural (Lee et al., 2017):** end-to-end con
  span representations. SOTA hasta transformers.
- **Transformer-based (Toshniwal et al., 2021):**
  fine-tuning de SpanBERT para coref. SOTA en OntoNotes.

**Métricas.**

- **MUC:** cuenta el mínimo de operaciones (insert,
  delete) para alinear las menciones predichas con las
  gold.
- **B³:** precisión y recall por mención individual.
- **CEAF:** F1 con mejor matching uno-a-uno entre
  clusters predichos y gold.
- **CoNLL F1:** promedio de MUC, B³, y CEAF F1.

**Aplicaciones.**

- **QA:** saber qué entidad se pregunta.
- **Resumen:** coherencia de las menciones a lo largo
  del texto.
- **Diálogo:** tracking de entidades en conversaciones
  multi-turno.
- **Extracción de información:** agrupar menciones
  múltiples en una sola entidad.

**Trampas.**

- **Ambigüedad pronominal:** "Juan vio a Pedro. Él
  sonrió." → ¿quién? Los modelos son malos en esto.
- **Pronombres neutros:** "ello" puede ser abstracto o
  concreto. El modelo necesita contexto amplio.
- **Cross-document coref:** en un corpus de múltiples
  documentos, ¿menciona "Apple" la misma entidad?

## Constrúyelo

```python
import re


def hobbs_algorithm(tokens, parse_tree, mention_pairs):
    """Algoritmo de Hobbs simplificado: encuentra el antecedente
    más cercano para cada pronombre."""
    resolved = []
    for token, pos in tokens:
        if pos == "PRON" and token.lower() in {"él", "ella", "ellos", "ellas"}:
            # Buscar el sustantivo más cercano antes
            for prev_token, prev_pos in reversed(resolved):
                if prev_pos in {"NOUN", "PROPN"}:
                    resolved.append((token, pos, prev_token))
                    break
            else:
                resolved.append((token, pos, None))
        else:
            resolved.append((token, pos, None))
    return resolved


def extract_mentions(text):
    """Extracción simple: pronombres y nombres propios."""
    mentions = []
    for match in re.finditer(r"\b(Él|Ella|Ellos|ellas|Juan|María|Pedro)\b", text):
        mentions.append((match.group(), match.start(), match.end()))
    return mentions


def cluster_mentions(mentions):
    """Clustering naive: agrupa menciones a la misma entidad
    basándose en la cadena string."""
    clusters = {}
    for m in mentions:
        # Simplificado: misma palabra = mismo cluster
        word = m[0].lower()
        clusters.setdefault(word, []).append(m)
    return clusters
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-coref
fase: 05
leccion: 24
---

Eres un asistente que ayuda con resolución de correferen-
cias. Recibirás el texto y la aplicación. Tu trabajo:

1. Si necesitas SOTA: SpanBERT o LingMess fine-tuneado
   en OntoNotes.
2. Para español: spaCy con es_core_news_trf.
3. Para producción rápida: reglas + modelo neuronal.
4. Para QA / diálogo: integrar con el pipeline de
   NLU principal.
5. Evaluar con MUC, B³, CEAF F1.
6. Advertir contra corref cross-document sin alineación
   explícita.
7. Manejar ambigüedades pronominales explícitamente.
```

## Ejercicios

1. **Hobbs**: implementa Hobbs para inglés.
2. **spaCy**: aplica corref a un texto en español.
3. **Desafío**: fine-tunea SpanBERT en OntoNotes.

## Lecturas recomendadas

- *Coreference Resolution* — Jurafsky & Martin.
- *SpanBERT* — Joshi et al., 2020.
- *LingMess* — Toshniwal et al., 2021.
- spaCy coref: <https://spacy.io/universe/project/coreferee>.

---

> 📚 **Adaptación al español** de la lección "[Coreference Resolution]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
