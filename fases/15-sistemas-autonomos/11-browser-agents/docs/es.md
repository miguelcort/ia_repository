# Browser agents

> Browser agents SOTA 2024-25: (1) Browser Use (Browser Use 2024 +OSS +LangChain +hybrid), (2) Anthropic Computer Use (Anthropic 2024 +API +claude-3-5-sonnet +screenshot), (3) OpenAI Operator (OpenAI 2025 +Product +computer-use-preview +managed), (4) Stagehand (Browserbase 2024 +OSS +Playwright +DOM), (5) Skyvern (Skyvern 2024 +OSS +workflows +hybrid), (6) Anchor (2024), (7) LaVague (2024). Approaches: DOM-based (Stagehand) +HTML+Playwright, screenshot-based (Computer Use, Operator) +Pixel, hybrid (Browser Use, Skyvern) +DOM+Pixel. Decision: DOM cuando estructura estable, screenshot cuando UI compleja, hybrid cuando mix. Patrón común: navigate (URL+page) -> screenshot (pixel+OCR) -> extract (DOM+text) -> act (click+type+submit) -> observe (state+loop) -> plan (steps+execute). Criterios: Browser Use -> OSS+LangChain, Computer Use -> API+Pixel+Claude, Operator -> Product+GPT+managed, Stagehand -> OSS+Playwright+DOM, Skyvern -> OSS+Workflows+Hybrid. Frameworks: langchain, playwright, anthropic, openai, browserbase. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + browser agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar BROWSER_AGENTS dict con 5+ agents.
- Implementar list_browser_agents + get_agent.
- Implementar filter_open_source + filter_with_workflows.
- Implementar plan_browser_task.
- Diagnosticar approaches.

## Constrúyelo

```python
BROWSER_AGENTS = {
    "browser_use": {
        "name": "Browser Use",
        "vendor": "Browser Use",
        "type": "OSS",
        "modalities": ["text", "screenshot"],
        "tools": ["click", "type", "scroll", "navigate"],
    },
    # ... 5+ agents
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
name: browser-agents
fase: 15
leccion: 11
---

1. Browser Use + Computer Use.
2. Operator + Stagehand.
3. Skyvern.
4. +Production.
```

## Ejercicios

1. **Browser agents**: probar
   diferentes browser agents.
2. **Plan**: descomponer una
   task en pasos.
3. **Desafio**: implementar
   un browser agent con
   Playwright.

## Lecturas recomendadas

- "Browser Use" (Browser Use, 2024)
- "Anthropic Computer Use" (Anthropic, 2024)
- "OpenAI Operator" (OpenAI, 2025)
- "Stagehand" (Browserbase, 2024)

---

> 📚 **Adaptación al español de la lección [Browser Agents]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).