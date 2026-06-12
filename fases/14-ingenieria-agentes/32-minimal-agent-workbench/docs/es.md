# Minimal agent workbench

> Minimal agent workbench: (1) Tool registry (register+dispatch), (2) Memory (add+get+last+clear), (3) Run loop (step+max), (4) Plan + execute (steps+run), (5) Verifier (check+validate), (6) Structured output (JSON+schema). Tool: name+fn+description+schema, call(*args, **kwargs). ToolRegistry: tools dict, register(tool) TypeError si no Tool, get(name), list(), dispatch(name, *args) get+call+KeyError. Memory: add(role, content) UUID+timestamp, get(id) dict+None, all() list, last(n) [-n:], clear() reset. MinimalWorkbench: step(tool, *args) memory+dispatch+increment, run() plan loop, max_steps limit+RuntimeError, reset memory+results. Criterios: Minimal = learning+prototype+simple, LangGraph = complex state+cycles+persistence, CrewAI = multi-agent+roles+crews. Decision: learn -> minimal, state -> LangGraph, multi-agent -> CrewAI, mix -> minimal + others. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + workbench.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/31
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar Tool con name + fn + description + schema.
- Implementar ToolRegistry con register + get + list + dispatch.
- Implementar Memory con add + get + all + last + clear.
- Implementar MinimalWorkbench con step + run + max_steps.
- Diagnosticar minimal vs frameworks.

## Constrúyelo

```python
class Tool:
    def __init__(self, name, fn, description="", schema=None):
        self.name = name
        self.fn = fn
        self.description = description
        self.schema = schema or {}

    def call(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: minimal-agent-workbench
fase: 14
leccion: 32
---

1. Tool + ToolRegistry.
2. Memory + MinimalWorkbench.
3. step + run + max_steps.
4. +Production.
```

## Ejercicios

1. **Tool**: probar
   register + dispatch.
2. **Memory**: probar
   add + last.
3. **Desafio**: integrar
   con LangGraph o CrewAI.

## Lecturas recomendadas

- "Anthropic: Building Effective Agents" (Anthropic, 2024)
- "LangGraph: StateGraph" (LangChain, 2024)
- "CrewAI: Multi-Agent" (CrewAI, 2024)

---

> 📚 **Adaptación al español de la lección [Minimal Agent Workbench]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).