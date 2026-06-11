# Tool schema design

> Tool schema design: +Specific +Clear name (snake_case verbo + objeto, lowercase, unique, -abbreviations, -generic), +Clear description (length 50-500, when-to-use, when-not-to-use, examples 1-3, edge cases, -TODO/-FIXME, -generic, +LLM-friendly, +specific, +unambiguous), +Min parameters (-noise), +Valid JSON schema (required, types, enums, constraints). +Insights: +quality, +reliability, -hallucinations, -tool misuse. Frameworks: openai, anthropic, langchain, MCP, FastMCP. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP. Trade-offs: detailed + quality, terse + simple. Lint: name validation (lowercase, snake_case, no leading/trailing underscore), description quality (length, when-to-use, examples, when-not, no TODO), conflicts (no duplicate names). Decision: simple -> minimal, production -> detailed, MCP -> server + client. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01, 13/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar score_description (heuristica).
- Implementar validate_name.
- Implementar check_name_conflict.
- Implementar build_tool_doc con when-to-use, when-not, examples.
- Implementar lint_tools.
- Diagnosticar naming y description best practices.

## Constrúyelo

```python
def score_description(description):
    score = 0.0
    if 50 <= len(description) <= 500:
        score += 0.3
    if "when" in description.lower() or "use" in description.lower():
        score += 0.2
    if "example" in description.lower():
        score += 0.2
    if "not" in description.lower():
        score += 0.2
    if not any(c in description for c in ["TODO", "FIXME"]):
        score += 0.1
    return min(score, 1.0)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: tool-schema
fase: 13
leccion: 05
---

1. snake_case name.
2. +Description quality.
3. +When-to-use +when-not.
4. +Examples 1-3.
5. Lint + tests.
```

## Ejercicios

1. **Lint tools**: lintear
   tools production.
2. **MCP schema**: diseñar
   schema para MCP server.
3. **Desafio**: tool
   generation agent.

## Lecturas recomendadas

- "OpenAI Function Calling Best Practices" (OpenAI Cookbook, 2024)
- "Anthropic Tool Use Guide" (Anthropic, 2024)
- "MCP Server Schema Design" (MCP, 2024)
- "JSON Schema Best Practices" (https://json-schema.org)

---

> 📚 **Adaptación al español de la lección [Tool Schema Design]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).