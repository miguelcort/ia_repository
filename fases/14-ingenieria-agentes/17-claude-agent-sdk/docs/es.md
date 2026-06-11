# Claude Agent SDK

> Claude Agent SDK (Anthropic 2024): build production agents con Claude. Components: (1) Agent (name + system_prompt + tools list + sub_agents list + model claude-3-5-sonnet + conversation), (2) BashTool (allowed_commands, run command +sandboxed), (3) FileSystemTool (root + files dict, read + write + list). Sub-agents: main delegates to sub-agent via "ask" keyword, sub specializes + return result. +Sub-agents, +Computer use, +Bash, +FileSystem, +Web, +Edit, +Tools, +Production, +Specialization, +Modular, +Composable, +Reliable, +Scalable. Variants: Claude seminal, sub-agents, computer use, tools, hierarchical, recursive. Frameworks: anthropic, claude, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, computer use, code, file, bash, web, edit, specialization. Decision: computer -> Claude, handoffs -> OpenAI, cycles -> LangGraph, code -> smolagents, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + Claude.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13, 14/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ClaudeAgent con sub-agents.
- Implementar BashTool.
- Implementar FileSystemTool.
- Diagnosticar Claude Agent SDK vs OpenAI Agents vs LangGraph.
- Diagnosticar sub-agents flow.

## Constrúyelo

```python
class ClaudeAgent:
    def run(self, input_text, max_turns=10):
        if self.sub_agents and "ask" in input_text.lower():
            sub = self.sub_agents[0]
            self.delegations += 1
            result = sub.run(f"Delegated: {input_text}")
            return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: claude-agent
fase: 14
leccion: 17
---

1. Sub-agents.
2. Tools (Bash, FileSystem).
3. Computer use.
4. +Production.
5. +Specialization.
```

## Ejercicios

1. **Claude Agent**: usar
   Claude Agent SDK con
   sub-agents.
2. **Bash**: implementar
   BashTool sandboxed.
3. **Desafio**: full
   computer use agent.

## Lecturas recomendadas

- "Claude Agent SDK" (Anthropic, 2024)
- "Building Agents with Claude" (Anthropic, 2024)
- "Sub-Agents Best Practices" (Anthropic, 2024)
- "Computer Use" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Claude Agent SDK]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).