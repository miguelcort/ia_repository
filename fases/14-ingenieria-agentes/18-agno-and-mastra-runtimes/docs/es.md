# Agno and Mastra runtimes

> Agno (Phidata 2024) y Mastra (2024): agent runtimes. Agno: (1) Agent (name + role + tools + memory + knowledge + reasoning + model), (2) Reasoning step-by-step + transparent + show trace + debugging, (3) Memory, (4) Knowledge, (5) Multi-agent. Mastra: (1) Agent (name + instructions + tools + workflows + model), (2) Workflows (execute named, sequential/parallel), (3) +Production +TypeScript-first. Agno: +Memory, +Knowledge, +Reasoning, +Multi-agent, +Transparent, +Quality, +Iterative, +Refined. Mastra: +Workflows, +Production, +TypeScript-first, +Reliable, +Scalable. Variants: Agno seminal, Mastra seminal, LangGraph, smolagents, custom. Frameworks: agno, phidata, mastra, langchain, smolagents, openai, anthropic. +Production: standard 2024-25. +Use cases: agent, multi-agent, reasoning, memory, knowledge, workflows. Decision: memory -> Agno, workflows -> Mastra, cycles -> LangGraph, code -> smolagents, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar AgnoAgent con tools/memory/knowledge/reasoning.
- Implementar MastraAgent con tools/workflows.
- Diagnosticar reasoning flow.
- Diagnosticar Agno vs Mastra trade-offs.

## Constrúyelo

```python
class AgnoAgent:
    def run(self, input_text):
        response_parts = [f"[{self.name}]"]
        if self.reasoning:
            response_parts.append("[Reasoning]")
        if self.memory:
            response_parts.append("[Memory used]")
        response_parts.append(input_text[:50])
        return " ".join(response_parts)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: agno-mastra
fase: 14
leccion: 18
---

1. Agno reasoning.
2. Mastra workflows.
3. Memory + knowledge.
4. +Production.
5. +Multi-agent.
```

## Ejercicios

1. **Agno**: usar Agno
   con reasoning + memory.
2. **Mastra**: probar Mastra
   con workflows.
3. **Desafio**: production
   runtime con Agno/Mastra.

## Lecturas recomendadas

- "Agno: Agent Runtime" (Phidata, 2024)
- "Mastra: TypeScript Agent Runtime" (Mastra, 2024)
- "Phidata Documentation" (Phidata, 2024)
- "Agent Runtimes Comparison" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Agno and Mastra Runtimes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).