# MCP roots and elicitation

> MCP roots: filesystem boundaries via file:// URIs only. Client provides via roots/list (returns URIs + name). Server checks within_root(path, root_uri) para sandbox. Security: +Boundaries, +Safe, -Escape. +Production: +Standardized, +Safe, +Modular. MCP elicitation: server asks user structured input via client via elicitation/create. Request: message + requestedSchema (JSON schema). Response: action (accept/decline/cancel) + content (structured values). +HITL, +Structured, +Validated, +Reliable, +Standardized, +Safe, +Modular. Frameworks: mcp, fastmcp, anthropic, openai, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE, sandbox, approval. Decision: security -> roots, user input -> elicitation, production -> ambos. Trade-offs: roots + security, elicitation + structured, freeform + simple. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_roots_list y validate_root_uri.
- Implementar within_root check.
- Implementar make_elicitation_request/response.
- Diagnosticar roots vs elicitation.
- Diagnosticar security boundaries.

## Constrúyelo

```python
def within_root(path, root_uri):
    if not validate_root_uri(root_uri):
        return False
    root_path = root_uri.replace("file://", "")
    return path.startswith(root_path + "/") or path == root_path
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-roots
fase: 13
leccion: 12
---

1. Roots: file:// boundaries.
2. Client provides.
3. within_root check.
4. Elicitation: user input.
5. +Security +HITL.
```

## Ejercicios

1. **Roots**: implementar
   filesystem sandbox.
2. **Elicitation**: agregar
   user input flow.
3. **Desafio**: custom
   approval flow.

## Lecturas recomendadas

- "MCP Roots Specification" (Anthropic, 2024)
- "MCP Elicitation Specification" (Anthropic, 2024)
- "Filesystem Sandboxing" (Anthropic, 2024)
- "Human-in-the-Loop" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Roots and Elicitation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).