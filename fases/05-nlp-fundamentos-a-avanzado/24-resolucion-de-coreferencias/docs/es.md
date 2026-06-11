# Resolución de coreferencias

> Agrupar menciones en entidades: 'Maria' = 'ella' = 'la nina'. SOTA: LingMess (84 F1), fastcoref (rapido, 80+ F1). Util para QA, summarization, dialogo. spaCy, fastcoref, coreferee como frameworks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 23-estrategias-de-chunking-y-rag
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar resolucion de coreferencias mock.
- Calcular similitud entre menciones.
- Diagnosticar cuando usar coref.
- Diagnosticar modelos SOTA.

## Constrúyelo

```python
def mention_pair_score(m1_text, m2_text, emb):
    return float(emb[m1_text] @ emb[m2_text] / (np.linalg.norm(emb[m1_text]) * np.linalg.norm(emb[m2_text])))
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

1. Default: fastcoref (rapido, GPU, 80+ F1).
2. Multilingual: spaCy coref.
3. Max accuracy: LingMess, SpanBERT fine-tune.
4. Custom: anotar OntoNotes, fine-tune SpanBERT.
5. LLM zero-shot: prompt con pronombres y opciones.
6. Combinar con entity linking.
```

## Ejercicios

1. **fastcoref**: instalar y aplicar fastcoref a un texto.
2. **OntoNotes schema**: anotar 1K docs en OntoNotes
   coref schema.
3. **Desafio**: pipeline completo de QA con
   coreference + RAG, mejorar accuracy en preguntas
   con pronombres.

## Lecturas recomendadas

- "End-to-end Coreference Resolution" (Joshi et al., 2019)
- "SpanBERT" (Joshi et al., 2020)
- fastcoref: <https://github.com/shon-otmazgin/fastcoref>
- coreferee: <https://github.com/vdobrovolschi/coreferee>

---

> 📚 **Adaptación al español** de la lección "[Coreference Resolution]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).