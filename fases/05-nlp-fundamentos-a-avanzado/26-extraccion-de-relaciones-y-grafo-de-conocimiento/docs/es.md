# Extracción de relaciones y grafo de conocimiento

> Detectar tripletas (head, relation, tail) en texto. TACRED, FewRel. Pipeline: NER + EL + RE -> KG. KG completion: TransE, RotatE. Almacenar: Neo4j, Wikidata. Aplicaciones: fact extraction, QA sobre KG, semantic search.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 25-vinculacion-de-entidades
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Extraer entidades mayusculas (NER mock).
- Extraer relaciones con regex.
- Clasificar tipo de relacion.
- Diagnosticar NER + EL + RE pipeline.

## Constrúyelo

```python
def extraer_relaciones_patron(texto):
    relaciones = []
    patrones = [
        (r"(\w+) (?:es presidente|fue presidente) de (\w+)", "presidente_de"),
        (r"(\w+) (?:trabaja en|vive en) (\w+)", "vive_en"),
    ]
    for patron, relacion in patrones:
        for match in re.finditer(patron, texto, re.IGNORECASE):
            relaciones.append((match.group(1), relacion, match.group(2)))
    return relaciones
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-relation-extraction
fase: 05
leccion: 26
---

1. Default: BERT fine-tune TACRED (70+ F1).
2. Few-shot: GPT-4/Claude con schema.
3. KG construction: NER + EL + RE -> Neo4j.
4. KG completion: TransE, RotatE, ComplEx.
5. QA sobre KG: LLM + Cypher/SPARQL.
6. Wikidata + Neo4j.
```

## Ejercicios

1. **BERT RE**: fine-tunear BERT en TACRED, alcanzar
   F1 > 70.
2. **Wikidata SPARQL**: query a Wikidata para verificar
   tripletas.
3. **Desafio**: pipeline completo IE + KG sobre un
   dominio custom, almacenar en Neo4j, QA con LLM +
   Cypher.

## Lecturas recomendadas

- "TACRED" (Zhang et al., 2017)
- "TransE" (Bordes et al., 2013)
- "RotatE" (Sun et al., 2019)
- Neo4j: <https://neo4j.com/>

---

> 📚 **Adaptación al español** de la lección "[Relation Extraction and KG]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).