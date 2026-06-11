# The agent loop

> The agent loop: foundation de todos los agent frameworks. Pattern: Think -> Act -> Observe -> Repeat. ReAct (Yao 2022) seminal. Components: (1) think (LLM reasoning), (2) act (tool call), (3) observe (result), (4) loop hasta finish o max_iterations, (5) history tracking. Variants: (1) ReAct (think-act-observe iterative), (2) ReWoo (plan-and-execute -observation +efficient), (3) Reflexion (verbal RL +self-reflection +memory), (4) ToT (tree of thoughts BFS/DFS +exploration), (5) LATS (language agent tree search +MCTS +planning), (6) Self-Refine (critic feedback +iterative +quality). +Iterative, +Reasoning, +Tools, +Planning, +Production. Frameworks: langchain, openai, anthropic, autogen, crewai, langgraph, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, automation, multi-agent. Decision: general -> ReAct, planning -> ReWoo, learning -> Reflexion, exploration -> ToT, search -> LATS, quality -> Self-Refine, production -> ReAct o ReWoo. 2025: +MCP + A2A + native + agent loop.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/16, 13/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar AgentLoop con think-act-observe.
- Implementar step y run.
- Implementar react_prompt.
- Implementar parse_react_output.
- Diagnosticar agent loop variants.

## Constrúyelo

```python
class AgentLoop:
    def step(self, observation):
        thought, action = self.llm_fn(observation, self.tools)
        if action is None:
            return thought, None
        name, args = action
        result = self.tools[name](**args)
        return thought, result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: agent-loop
fase: 14
leccion: 01
---

1. Think-act-observe.
2. ReAct pattern.
3. Max iterations.
4. History.
5. +Production.
```

## Ejercicios

1. **ReAct**: implementar
   ReAct custom con
   HuggingFace.
2. **LangChain**: usar
   LangChain AgentExecutor.
3. **Desafio**: full
   agent production.

## Lecturas recomendadas

- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "LangChain Agents" (LangChain Documentation)
- "OpenAI Agents SDK" (OpenAI, 2024)
- "Anthropic Building Effective Agents" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [The Agent Loop]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).