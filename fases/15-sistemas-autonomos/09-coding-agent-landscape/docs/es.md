# Coding agent landscape

> Coding agent landscape SOTA 2024-25: (1) Devin (Cognition 2024 +autonomous +E2E), (2) Cursor (Cursor 2023 +IDE), (3) GitHub Copilot (2021 +IDE), (4) Claude Code (Anthropic 2024 +CLI), (5) Codex (OpenAI 2021 +CLI/API), (6) Aider (2023 +CLI +open source), (7) Cline (2024 +VS Code +open source), (8) Windsurf (Codeium 2024 +IDE), (9) Bolt.new (StackBlitz 2024 +Web IDE), (10) v0 (Vercel 2023 +UI gen). Architectures: autonomous (Devin), IDE-integrated (Cursor, Copilot, Windsurf), CLI (Claude Code, Aider, Codex), Web IDE (Bolt), UI generator (v0). Open source: Aider (CLI), Cline (VS Code), OpenHands (autonomous). Tools: bash, read, write, edit, search, browser, computer, linter, deploy. +Production, +Tools, +Specialized, +Reliable, +Standard. Frameworks: anthropic, openai, codeium, stackblitz, vercel, cognition, aider, cline. +Production: standard 2024-25. +Use cases: coding, IDE, CLI, web, autonomous, UI. Decision: autonomous -> Devin, IDE -> Cursor, CLI -> Claude Code o Aider, UI -> v0, open source -> Aider o Cline, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar CODING_AGENTS dict con 10+ agents.
- Implementar list_coding_agents + get_agent.
- Implementar filter_agents con criteria.
- Diagnosticar architectures.
- Diagnosticar open source vs commercial.

## Constrúyelo

```python
CODING_AGENTS = {
    "devin": {
        "name": "Devin",
        "vendor": "Cognition",
        "type": "autonomous",
        "tools": ["shell", "browser", "editor", "search"],
        "release_year": 2024,
        "open_source": False,
    },
    # ... 10+ agents
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
name: coding-agents
fase: 15
leccion: 09
---

1. Devin, Cursor, Claude Code.
2. Codex, Aider, Cline.
3. Windsurf, Bolt, v0.
4. +Open source +Production.
5. +Tools.
```

## Ejercicios

1. **Coding agents**: probar
   diferentes coding agents.
2. **Open source**: usar
   Aider o Cline.
3. **Desafio**: comparar
   coding agents en benchmark.

## Lecturas recomendadas

- "Devin: AI Software Engineer" (Cognition, 2024)
- "Claude Code" (Anthropic, 2024)
- "Aider: AI Pair Programming" (Aider, 2023)
- "Cline: VS Code AI Agent" (Cline, 2024)

---

> 📚 **Adaptación al español de la lección [Coding Agent Landscape]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).