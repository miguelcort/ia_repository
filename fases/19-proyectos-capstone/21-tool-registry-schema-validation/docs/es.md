# 21 — Tool registry y schema validation

> Tool registry: catálogo central de tools con schemas (JSON Schema), permisos (allow/deny), rate limits, rate tracking. Validación: pydantic, JSON Schema. MCP standard. Frameworks: LangChain, FastMCP.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 13, Fase 19/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Tool registry con schemas.
- Validation (pydantic, JSON Schema).
- Permission system.
- Rate limiting.

## El problema

Tool registry es la fuente de verdad: cada tool
tiene (1) name, (2) description (para LLM), (3)
JSON Schema (input validation), (4) permissions
(allow/deny por tool, per agent), (5) rate limits
(N calls per minute). Validación: pydantic para
Python, JSON Schema para cross-language. MCP usa
JSON Schema. Errores de validación: log + refuse
+ retry prompt.

## Constrúyelo

```python
from pydantic import BaseModel, ValidationError
import jsonschema


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn, schema, permissions):
        self.tools[name] = {"fn": fn, "schema": schema,
                           "permissions": permissions}

    def execute(self, name, args, agent_id):
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        if not self._check_permission(tool, agent_id):
            raise PermissionError(f"Denied: {name}")
        try:
            jsonschema.validate(args, tool["schema"])
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid args: {e}")
        return tool["fn"](**args)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tool-registry
fase: 19
leccion: 21
---

1. Schema por tool.
2. Permission system.
3. Rate limiting.
4. JSON Schema validation.
5. MCP-compatible.
```

## Ejercicios

1. **Registry**: 10 tools
   con schemas.
2. **Validation**: invalid
   args handling.
3. **Desafío**: rate limiting
   per agent.

## Lecturas recomendadas

- "MCP" (Anthropic 2024)
- "JSON Schema" (2024)
- "Pydantic" (2024)
- "FastMCP" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[21-tool-registry-schema-validation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
