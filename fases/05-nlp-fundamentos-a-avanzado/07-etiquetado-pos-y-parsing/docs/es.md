# Etiquetado POS y parsing

> Asignar categoria gramatical (POS) y estructura sintactica (parsing). Base de NER, sentiment, IE, MT. spaCy y Stanza son los frameworks default. Universal Dependencies es el esquema estandar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-reconocimiento-de-entidades
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar POS tagger mock.
- Implementar dependency parser mock.
- Diagnosticar constituency vs dependency.
- Diagnosticar modelos SOTA.

## Constrúyelo

```python
def pos_tag_mock(tokens):
    det = {"el", "la", "los", "las"}
    verbos = {"come", "salta", "ladra"}
    nombres = {"gato", "perro", "nino"}
    tags = []
    for t in tokens:
        if t in det: tags.append("DET")
        elif t in verbos: tags.append("VERB")
        elif t in nombres: tags.append("NOUN")
        else: tags.append("NOUN")
    return tags
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pos-dep
fase: 05
leccion: 07
---

1. Espanol: spaCy es_core_news_lg.
2. Alta accuracy: Stanza.
3. Multilingual: UDPipe, Trankit.
4. Custom: BERT fine-tune en UD.
5. Visualizar: displacy.
```

## Ejercicios

1. **HMM POS tagger**: implementar Viterbi para POS con
   emisiones y transiciones.
2. **Biaffine**: implementar dependency parser con
   biaffine attention.
3. **Desafio**: fine-tunear Stanza o spaCy en un dominio
   custom (e.g. recetas de cocina en espanol).

## Lecturas recomendadas

- "Universal Dependencies" (Nivre et al., 2016)
- "Stanza" (Qi et al., 2020)
- spaCy: <https://spacy.io/>

---

> 📚 **Adaptación al español** de la lección "[POS Tagging and Parsing]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).