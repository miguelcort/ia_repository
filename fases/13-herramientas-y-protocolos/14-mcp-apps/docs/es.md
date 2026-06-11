# MCP apps

> MCP apps: apps dentro de MCP con UI components y app-like experiences en chat. Components: text (content), image (url + alt), button (label + action + style primary/secondary/danger), form (fields + submitLabel), chart (chartType + data + options). Manifest: name, version, description, capabilities, components. render_app: text -> rendered, button: action handler, form: render fields, chart: render visual. +UI, +Interactive, +Rich, +Standardized, +UX, +Reusable, +Modular, +Testable. Frameworks: mcp, fastmcp, anthropic, openai, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation, IDE, dashboards, chat. Decision: rich UX -> apps, simple -> text, production -> apps. Trade-offs: cada component + specialty, rich + UX, simple + naive. Hoy: SOTA 2024-25 mix. 2025: +MCP + A2A + native + rich UI.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_app_manifest.
- Implementar make_text/image/button/form/chart_component.
- Implementar render_app con action handlers.
- Diagnosticar UI components.
- Diagnosticar apps vs simple text.

## Constrúyelo

```python
def make_button_component(label, action, style="primary"):
    return {"type": "button", "label": label, "action": action, "style": style}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-apps
fase: 13
leccion: 14
---

1. Manifest.
2. UI components.
3. Render.
4. Action handlers.
5. +UI +Rich.
```

## Ejercicios

1. **MCP app**: construir
   custom app con form.
2. **Chart**: agregar
   chart component.
3. **Desafio**: dashboard
   con multiple components.

## Lecturas recomendadas

- "MCP Apps Specification" (Anthropic, 2024)
- "Building MCP Apps" (Anthropic Cookbook, 2024)
- "ChatGPT Apps SDK" (OpenAI, 2024)
- "Claude Artifacts" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Apps]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).