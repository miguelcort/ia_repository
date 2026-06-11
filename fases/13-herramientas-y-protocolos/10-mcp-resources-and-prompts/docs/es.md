# MCP resources and prompts

> MCP resources: read-only data exposure via URI scheme (file:// local, https:// remote, db:// database, custom), mime types (text/plain, text/markdown, application/json, image/png), resources/list + resources/read. Security: URI validation, sandboxing, auth. +Reusable, +Modular, +Standardized, +Safe, +Caching, +Pagination, +Auth. MCP prompts: reusable templates con arguments (name, description, required). Argument spec, render: replace {arg} con values (-Missing -> empty string). prompts/list + prompts/get. +Reusable, +Modular, +Standardized, +Maintainable, +Caching, +Validation, +Composition. Variants: resources, prompts, tools, sampling, roots, elicitation. Frameworks: mcp, fastmcp, anthropic, openai, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE. Decision: data -> resources, templates -> prompts, actions -> tools, production -> combinacion. Trade-offs: resources/prompts + reusable, freeform + simple. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar resource_uri y validate_resource_uri.
- Implementar make_resource con mime type.
- Implementar resource_read.
- Implementar make_prompt y render_prompt.
- Diagnosticar resources vs prompts vs tools.

## Constrúyelo

```python
def render_prompt(prompt, args=None):
    args = args or {}
    rendered = prompt["description"]
    for arg in prompt.get("arguments", []):
        rendered = rendered.replace(f"{{{arg['name']}}}", str(args.get(arg["name"], "")))
    return {"description": rendered, "messages": [{"role": "user", "content": rendered}]}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-resources-prompts
fase: 13
leccion: 10
---

1. Resources read-only.
2. URI scheme + mime.
3. Prompts templates.
4. Render con args.
5. +Reusable.
```

## Ejercicios

1. **Resources**: exponer
   filesystem via MCP.
2. **Prompts**: crear
   prompts library.
3. **Desafio**: custom
   URI scheme.

## Lecturas recomendadas

- "MCP Resources Specification" (Anthropic, 2024)
- "MCP Prompts Specification" (Anthropic, 2024)
- "Building Reusable Prompts" (Anthropic Cookbook, 2024)
- "MIME Types Reference" (IANA, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Resources and Prompts]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).