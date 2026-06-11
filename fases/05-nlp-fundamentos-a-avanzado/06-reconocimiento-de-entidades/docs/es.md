# Reconocimiento de entidades (NER)

> Detectar y clasificar entidades nombradas: personas, lugares, organizaciones, fechas, dinero. Base de extraccion de informacion, busqueda, knowledge graphs. BERT fine-tune es el SOTA en CoNLL-2003 (~94% F1).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-analisis-de-sentimiento
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar esquema BIO tagging.
- Extraer entidades de tags BIO.
- Evaluar con span F1.
- Diagnosticar modelos (CRF, BiLSTM, BERT, nested).

## Constrúyelo

```python
def extraer_entidades(tokens, tags):
    entidades = []
    actual_tipo = None
    inicio = None
    for i, (t, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if actual_tipo is not None:
                entidades.append((actual_tipo, (inicio, i-1), " ".join(tokens[inicio:i])))
            actual_tipo = tag[2:]
            inicio = i
        elif tag == "O":
            if actual_tipo is not None:
                entidades.append((actual_tipo, (inicio, i-1), " ".join(tokens[inicio:i])))
                actual_tipo = None
    return entidades
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ner
fase: 05
leccion: 06
---

1. Espanol: spaCy es_core_news_lg.
2. Custom: BERT fine-tune token classification.
3. Multilingual zero-shot: GLiNER.
4. Few-shot: GPT-4 con ejemplos.
5. Medical: BioBERT, SciSpaCy.
6. Anotar 1-10K, 3-5 epocas, lr 5e-5.
```

## Ejercicios

1. **CRF basico**: implementar NER con sklearn-crfsuite.
2. **GLiNER zero-shot**: usar GLiNER con tipos
   custom sin entrenar.
3. **Desafio**: BERT fine-tune en CoNLL-2002 espanol
   (anotar subset), alcanzar F1 > 0.85.

## Lecturas recomendadas

- "Introduction to the CoNLL-2003 Shared Task" (Tjong Kim Sang
  & De Meulder, 2003)
- "Design Challenges and Misconceptions in Named Entity
  Recognition" (Bender, 2014)
- GLiNER: <https://github.com/urchade/GLiNER>

---

> 📚 **Adaptación al español** de la lección "[Named Entity Recognition]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).