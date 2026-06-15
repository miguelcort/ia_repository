# 25 — Entity linking y desambiguación

> Entity linking: mapear menciones en texto a entidades en una knowledge base (Wikipedia, Wikidata). "Apple" puede ser la compañía, la fruta, o el nombre propio.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 24-resolucion-de-coreferencias
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar entity linking con candidate generation +
  ranking.
- Aplicar modelos pre-entrenados (BLINK, GENRE).
- Diagnosticar la ambigüedad de entidades.
- Conectar a grafos de conocimiento.

## El problema

NER extrae menciones ("Apple", "Madrid"). Entity linking
las conecta a entidades específicas en una knowledge base
(Q5 = Apple Inc., Q3957 = Madrid, Q89 = Apple fruit).
Resolver la ambigüedad ("Apple" = compañía, fruta, o
nombre) es el desafío. Es la base de grafos de
conocimiento, sistemas de recomendación, y chatbots
informados.

## El concepto

**Pipeline estándar.**

1. **NER:** extraer menciones.
2. **Candidate generation:** para cada mención,
   generar candidatos en la KB (e.g. todas las entidades
   cuyo nombre matchea "Apple").
3. **Disambiguation:** rankear los candidatos y elegir el
   correcto.
4. **Linking:** anotar la mención con la entidad.

**Candidate generation.** Opciones:

- **Surface matching:** todas las entidades cuyo
  nombre o alias matchea la mención. BM25 sobre el
  nombre.
- **Embeddings:** encoding bi-encoder (mención,
  entity) entrenado con contrastive loss. Top-K por
  cosine similarity.
- **Cross-encoder:** más lento pero más preciso.
  Re-rankea los top-K del bi-encoder.

**Modelos modernos.**

- **BLINK (Wu et al., 2020):** bi-encoder + cross-encoder
  para entity linking zero-shot. Pre-entrenado en
  Wikipedia.
- **GENRE (De Cao et al., 2021):** entity linking como
  generación seq2seq. Produce el nombre de la entidad
  como texto.
- **ReFinED (Ayoola et al., 2022):** combina NER +
  entity disambiguation en un solo modelo.

**Knowledge bases.**

- **Wikipedia:** la más grande, ~6M entidades en inglés.
- **Wikidata:** estructurada, con tipos y relaciones.
  Base de muchos KGs modernos.
- **Domain-specific:** UMLS (medicina), GeoNames
  (geografía), MusicBrainz (música).

**Métricas.**

- **Accuracy@1:** ¿la entidad top-1 es la correcta?
- **Recall@K:** ¿la correcta está en los top-K?
- **Macro F1:** promedio sobre entidades.

**Aplicaciones.**

- **Construcción de grafos de conocimiento:** extraer
  entidades y relaciones de texto.
- **Sistemas de recomendación:** producto, marca,
  categoría.
- **Chatbots informados:** resolver entidades para
  acceder a datos estructurados.
- **Fact-checking:** verificar claims contra una KB.

**Trampas.**

- **Entidades nuevas:** la KB no contiene toda entidad
  posible. Marcar como "NIL" o crear nueva.
- **Menciones ambiguas:** "Apple" puede ser 5+ entidades.
  El cross-encoder ayuda pero no resuelve todo.
- **Knowledge base desactualizada:** las entidades
  cambian. Usar versiones actualizadas.

## Constrúyelo

```python
import re
from collections import defaultdict


def candidate_generation(mention, kb):
    """Genera candidatos para una mención usando matching
    de superficie."""
    candidates = []
    mention_lower = mention.lower()
    for entity_id, entity in kb.items():
        names = [entity["name"]] + entity.get("aliases", [])
        for name in names:
            if name.lower() == mention_lower:
                candidates.append((entity_id, 1.0))
                break
            elif mention_lower in name.lower():
                candidates.append((entity_id, 0.5))
    return candidates


def disambiguate(mention, context, candidates, kb):
    """Ranking simple: prefiere entidades que co-ocurren
    en el contexto."""
    context_words = set(context.lower().split())
    scores = []
    for entity_id, base_score in candidates:
        entity_words = set(
            kb[entity_id]["name"].lower().split() +
            " ".join(kb[entity_id].get("aliases", [])).lower().split()
        )
        overlap = len(context_words & entity_words)
        scores.append((entity_id, base_score + overlap * 0.1))
    return sorted(scores, key=lambda x: -x[1])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-entity-linking
fase: 05
leccion: 25
---

Eres un asistente que ayuda con entity linking. Recibirás
el texto y la KB. Tu trabajo:

1. Si quieres SOTA: BLINK o GENRE pre-entrenado.
2. Para español: usar Wikidata en español.
3. Para dominio específico: construir KB propia.
4. Pipeline: NER → candidate generation → cross-encoder
   re-ranking.
5. Para entidades nuevas: marcar como NIL.
6. Evaluar con Accuracy@1.
7. Advertir contra ambigüedad sin contexto suficiente.
8. Logging: anotar las menciones no resueltas para
   revisión.
```

## Ejercicios

1. **Candidate generation**: implementa BM25 sobre los
   nombres de las entidades.
2. **Cross-encoder**: usa un modelo pre-entrenado
   para re-ranking.
3. **Desafío**: implementa entity linking para un
   dominio médico con UMLS.

## Lecturas recomendadas

- *BLINK* — Wu et al., 2020.
- *GENRE* — De Cao et al., 2021.
- *ReFinED* — Ayoola et al., 2022.
- Wikidata: <https://www.wikidata.org>.
- spaCy entity linker: <https://github.com/egerber/spaCy-entity-linker>.

---

> 📚 **Adaptación al español** de la lección "[Entity Linking and Disambiguation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
