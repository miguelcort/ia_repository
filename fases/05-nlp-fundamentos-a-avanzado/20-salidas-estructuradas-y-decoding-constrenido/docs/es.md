# 20 — Salidas estructuradas y decoding restringido

> Forzar al LLM a producir JSON válido, un esquema específico, o seguir un formato. Es la base de function calling, tool use, y agentes.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-ingenieria-llms
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar JSON mode y grammar-constrained decoding.
- Diagnosticar el trade-off entre flexibilidad y
  restricciones.
- Diseñar schemas para function calling.
- Usar Outlines, Guidance, y JSON schema en producción.

## El problema

Los LLMs son probabilísticos: producen cualquier token.
Para integrarlos en pipelines, necesitas que produzcan
formatos específicos: JSON válido, un esquema concreto,
SQL válido, código que compile, etc. Las salidas
estructuradas resuelven esto: guiar al LLM para que
produzca exactamente lo que necesitas. La lección cubre
las técnicas canónicas y las herramientas.

## El concepto

**JSON mode.** Forzar al LLM a producir JSON válido.
Algunos modelos (OpenAI, Gemini) tienen `response_format:
{"type": "json_object"}` nativo. Otros requieren prompts
cuidadosos o técnicas más avanzadas.

**Schema-constrained decoding.** Definir un esquema
(JSON Schema, Pydantic, Zod) y forzar al LLM a producir
tokens que solo generen outputs válidos. El LLM solo puede
elegir entre los tokens que mantienen la validez del
schema en cada paso.

**Grammar-constrained decoding.** Generalización:
usar una gramática formal (JSON, SQL, Python) y forzar al
LLM a producir solo secuencias que la gramática acepte.
Más expresivo que JSON schema.

**Tool use / Function calling.** El LLM produce una
llamada a función estructurada:

```json
{
  "name": "get_weather",
  "arguments": {"city": "Madrid"}
}
```

El sistema ejecuta la función y devuelve el resultado al
LLM. Es la base de los agentes modernos.

**Herramientas.**

- **Outlines:** library Python que hace schema-
  constrained decoding sobre cualquier LLM (vía
  token masking en el logit processor).
- **Guidance (Microsoft):** programación con templates
  y grammar-constrained generation.
- **LMQL:** declarative prompting con restricciones SQL-
  like.
- **Instructor (Python):** extrae objetos Pydantic de
  las salidas de LLM.
- **JSON schema + OpenAI function calling:** built-in.

**Cuándo usar cada técnica.**

| Necesidad | Herramienta |
|---|---|
| JSON simple | OpenAI `response_format` |
| JSON schema complejo | Outlines o Instructor |
| Gramática libre (Python, SQL) | Outlines o Guidance |
| Tool calling | OpenAI function calling |
| Solo post-procesar | Instructor + retry |

**Trampas.**

- **El LLM miente en el schema:** schema-constrained
  garantiza formato, no corrección semántica. Validar
  el output.
- **Schemas muy restrictivos:** el LLM no puede
  generar respuesta. Empezar permisivo y restringir.
- **No manejar errores:** parseo puede fallar. Siempre
  tener un retry con feedback.

## Constrúyelo

```python
import json
import re


def extract_json(text):
    """Extrae el primer JSON válido de un texto."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def validate_schema(data, schema):
    """Validador minimal: verifica campos requeridos y tipos."""
    for field, field_type in schema.items():
        if field not in data:
            return False, f"Campo requerido '{field}' falta"
        if not isinstance(data[field], field_type):
            return False, f"Campo '{field}' debe ser {field_type.__name__}"
    return True, "OK"


def function_call_schema():
    """Schema para function calling."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"]
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
name: prompt-structured-output
fase: 05
leccion: 20
---

Eres un asistente que ayuda a configurar salidas estruc-
turadas. Recibirás el caso de uso y el LLM. Tu trabajo:

1. Si el LLM soporta JSON mode nativo: usar
   `response_format`.
2. Si necesitas un schema complejo: Outlines o
   Instructor.
3. Si necesitas grammar completa: Guidance.
4. Si es function calling: OpenAI/Anthropic tool
   use.
5. Prompt: dar un ejemplo del JSON esperado.
6. Validar el output con Pydantic o Zod.
7. Manejar errores con retry y feedback.
8. Logging: siempre loguear el output crudo, no solo
   el parseado.
```

## Ejercicios

1. **JSON mode**: implementa un wrapper que fuerza
   JSON válido con retry.
2. **Function calling**: implementa un LLM con
   tool use con dos funciones.
3. **Desafío**: usa Outlines con un schema Pydantic
   para extracción estructurada.

## Lecturas recomendadas

- *JSON mode* — OpenAI docs.
- *Outlines* — Willard & Louf, 2023.
- *Guidance* — Microsoft Research.
- *Instructor* — <https://github.com/jxnl/instructor>.
- Pydantic: <https://docs.pydantic.dev>.

---

> 📚 **Adaptación al español** de la lección "[Structured Outputs and Constrained Decoding]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
