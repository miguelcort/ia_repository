# Vinculación de entidades

> Vincular menciones a entidades en una KB (Wikidata, Wikipedia). Pipeline: NER -> candidate generation (BM25/dense) -> disambiguation (cross-encoder). SOTA: BLINK, GENRE, ReFinED (88-89% AIDA). Aplicaciones: KG construction, fact extraction, semantic search.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 24-resolucion-de-coreferencias
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Construir KB mock.
- Implementar entity linking con exact + partial match.
- Diagnosticar NER vs EL.
- Diagnosticar modelos SOTA.

## Constrúyelo

```python
def entity_linking(mention, kb):
    for qid, ent in kb.entidades.items():
        if mention.lower() == ent["name"].lower():
            return qid
        if mention.lower() in [a.lower() for a in ent.get("aliases", [])]:
            return qid
    return None
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

1. Default: NER + BM25/dense + cross-encoder (BLINK).
2. Multilingual: mReFinED, mGENRE.
3. Custom KB: fine-tune dense bi-encoder.
4. Low-latency: ReFinED end-to-end.
5. LLM: Claude con KB en prompt.
6. Wikidata 100M+ entidades.
```

## Ejercicios

1. **BLINK**: usar BLINK para linking de entidades en
   español.
2. **Wikidata query**: SPARQL sobre Wikidata para
   verificar aliases.
3. **Desafio**: pipeline EL production con Wikidata
   subset, ReFinED, y coherence scoring.

## Lecturas recomendaciones

- "BLINK" (Wu et al., 2020)
- "GENRE" (De Cao et al., 2021)
- "ReFinED" (Ayoola et al., 2022)
- Wikidata: <https://www.wikidata.org/>

---

> 📚 **Adaptación al español** de la lección "[Entity Linking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).