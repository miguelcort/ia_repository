# Salidas estructuradas y decoding constreñido

> Forzar al LLM a generar output que satisface un schema (JSON, regex, gramática). Outlines, JSONformer, Guidance, OpenAI JSON mode, llama.cpp GBNF. Garantiza 100% output valido.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 19-tokenizacion-de-subpalabras
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar parser JSON simple.
- Validar contra schema.
- Implementar grammar-constrained decoding.
- Diagnosticar structured vs prompt+parsing.

## Constrúyelo

```python
def grammar_constrained_decoding(tokens_validos, logits):
    mask = np.full_like(logits, -1e9)
    for t in tokens_validos:
        if 0 <= t < len(mask):
            mask[t] = logits[t]
    return mask
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-structured-gen
fase: 05
leccion: 20
---

1. JSON APIs: Outlines, Instructor, OpenAI JSON mode.
2. Function calling: OpenAI tools, Anthropic.
3. Code: Codex, Code Llama constrained.
4. SQL: text-to-SQL con grammar.
5. Regex: Outlines regex, JSONformer.
6. Pydantic + retry on error.
```

## Ejercicios

1. **Pydantic**: definir schema con Pydantic para
   extraccion de info.
2. **Outlines**: usar Outlines para JSON-constrained
   generation con LLaMA.
3. **Desafio**: pipeline de extraccion de info
   production-ready con schema Pydantic, retry, y
   validacion.

## Lecturas recomendadas

- "JSON Structured Output" (OpenAI, 2024)
- "Outlines" (Willard & Louf, 2023)
- Instructor: <https://github.com/jxnl/instructor>

---

> 📚 **Adaptación al español** de la lección "[Structured Outputs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).