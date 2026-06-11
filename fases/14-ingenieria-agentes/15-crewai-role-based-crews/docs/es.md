# CrewAI role based crews

> CrewAI (2024): role-based multi-agent framework. Components: (1) Agents (role + goal + backstory + tools + llm, +outputs), (2) Tasks (description + agent + expected_output + context list of upstream tasks, +output), (3) Crews (agents + tasks + process sequential/parallel/hierarchical, kickoff). Processes: (1) sequential (task1 -> task2 +composition +context +dependency), (2) parallel (concurrent +speed +independent -context), (3) hierarchical (manager + workers +coordination). +Role-based, +Multi-agent, +Production, +Composition, +Reliable, +Scalable. Variants: CrewAI seminal, sequential, parallel, hierarchical, custom. Frameworks: crewai, langchain, openai, smolagents. +Production: standard 2024-25. +Use cases: agent, multi-agent, research, automation, role. Decision: role -> CrewAI, conversational -> AutoGen, cycles -> LangGraph, handoffs -> OpenAI Agents, production -> combinacion. Trade-offs: cada uno + specialty, sequential + composition, parallel + speed. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13, 14/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar CrewAIAgent (role + goal + backstory + tools).
- Implementar CrewAITask (description + agent + context).
- Implementar Crew con sequential/parallel.
- Diagnosticar CrewAI vs AutoGen vs LangGraph.
- Diagnosticar processes trade-offs.

## Constrúyelo

```python
class CrewAIAgent:
    def execute(self, task):
        output = f"[{self.role}] completed: {task.description[:50]}"
        self.outputs.append(output)
        return output
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: crewai
fase: 14
leccion: 15
---

1. Role-based agents.
2. Tasks.
3. Sequential/parallel.
4. Hierarchical.
5. +Production.
```

## Ejercicios

1. **CrewAI**: usar CrewAI
   con sequential crew.
2. **Parallel**: probar
   parallel execution.
3. **Desafio**: research
   crew con CrewAI.

## Lecturas recomendadas

- "CrewAI: Role-Based Multi-Agent Framework" (CrewAI, 2024)
- "CrewAI Documentation" (https://docs.crewai.com)
- "Sequential vs Parallel Crews" (CrewAI, 2024)
- "Building Multi-Agent Systems" (CrewAI, 2024)

---

> 📚 **Adaptación al español de la lección [CrewAI Role Based Crews]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).