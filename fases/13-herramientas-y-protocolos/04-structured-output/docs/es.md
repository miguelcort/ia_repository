# Structured output

> Structured output: LLM output con JSON schema enforced. OpenAI response_format ({type: json_schema, json_schema: {name, schema, strict: true}}), Pydantic models (Type-safe), Instructor library (client.chat.completions.create(response_model=PydanticModel), +retry +validation), Outlines (constrained decoding via FSM/grammar, +reliability -sampling freedom), Guidance (Microsoft, +tokens), lm-format-enforcer (+JSON schema +regex), JSON mode (legacy type: json_object -strict), Anthropic tool use (+structured). Frameworks: openai, anthropic, instructor, outlines, guidance, lm-format-enforcer, transformers. +Insights: +validated +reliable +type-safe -freeform. +Production: standard 2024-25. +Use cases: agent, RAG, extraction, classification, MCP. Trade-offs: structured + safe, freeform + flexible. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01, 13/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar json_schema_for_response.
- Implementar json_mode (legacy).
- Implementar parse_strict_json.
- Implementar pydantic_to_schema.
- Implementar instructor_wrap y outlines_constrained_decode.
- Diagnosticar response_format vs instructor vs outlines.

## Constrúyelo

```python
def json_schema_for_response(schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("name", "response"),
            "schema": schema.get("schema", {}),
            "strict": True,
        },
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: structured-output
fase: 13
leccion: 04
---

1. response_format.
2. Pydantic.
3. Instructor.
4. Outlines FSM.
5. +Validated.
```

## Ejercicios

1. **Instructor**: usar
   instructor con Pydantic.
2. **Outlines**: probar
   outlines constrained.
3. **Desafio**: structured
   extraction custom.

## Lecturas recomendadas

- "OpenAI Structured Outputs" (OpenAI, 2024)
- "Instructor: Structured LLM Outputs" (Liu, 2023)
- "Outlines: Guided Text Generation" (Willard et al., 2023)
- "Guidance: Controllable Generation" (Microsoft, 2023)

---

> 📚 **Adaptación al español de la lección [Structured Output]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).