# Outputs estructurados

> Structured outputs: JSON mode (OpenAI, Anthropic), JSON Schema, Pydantic/Zod validation, function calling (tool use, JSON {name, args}), guided generation con token masking (Outlines, Guidance, LMQL, Instructor). Frameworks: Pydantic-AI, Instructor, Outlines, Guidance, LangChain tools, MCP (Model Context Protocol). Beneficios: +reliability, -parsing errors, +type safety, -silent failures. Pipeline: LLM → parse → validate (Pydantic) → retry on fail. Hoy: structured outputs + function calling + MCP es production standard. Frontier: structured agentic LLMs con reasoning + tools + verification.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/01-prompt-engineering
**Tiempo estimado:** ~30 minutos

## Objetivos

- Extraer JSON de output LLM.
- Validar contra JSON Schema.
- Implementar function calling format.
- Diagnosticar guided generation.

## Constrúyelo

```python
def json_schema_validate(obj, schema):
    if not isinstance(obj, dict): return False
    for key, t in schema.items():
        if key not in obj: return False
        if not isinstance(obj[key], t): return False
    return True
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: structured-outputs
fase: 11
leccion: 03
---

1. JSON Schema, Pydantic.
2. Function calling.
3. Guided generation.
4. Outlines, Guidance.
5. MCP standard.
```

## Ejercicios

1. **JSON**: implementar parser
   con Pydantic.
2. **Function calling**: definir
   3 tools, ejecutar.
3. **Desafio**: Outlines con
   JSON Schema custom.

## Lecturas recomendadas

- "Function Calling and Other API Updates" (OpenAI, 2023)
- "JSON Mode" (OpenAI, Anthropic)
- "Outlines: Guided Generation" (575, 2023)
- "Model Context Protocol (MCP)" (Anthropic, 2024)
- "Pydantic AI" (Python, 2024)

---

> 📚 **Adaptación al español** de la lección "[Structured Outputs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).