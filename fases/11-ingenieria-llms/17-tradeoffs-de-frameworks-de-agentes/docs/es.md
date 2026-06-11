# Tradeoffs de frameworks de agentes

> Frameworks agentic SOTA 2024-25: LangGraph (stateful, state machines, LangSmith, PostgresSaver), LangChain (general, +ecosystem, LCEL), LlamaIndex (RAG, query engines, LlamaHub), DSPy (programmatic prompts, optimization, BootstrapFewShot, MIPRO), Letta (memory + agents), smolagents (HuggingFace, minimalist, CodeAgents, code-based), Mastra (TypeScript-first, MCP), OpenAI Agents SDK (OpenAI-specific, sessions), AutoGen (Microsoft), CrewAI (multi-agent). Combinaciones: LangGraph + LangSmith (state + observability), LlamaIndex + LangGraph (RAG + state), Letta + LangGraph (memory + state), MCP + cualquier (standard tools). Decision matrix: stateful → LangGraph, RAG → LlamaIndex, TypeScript → Mastra, memory → Letta, minimal → smolagents, prompt opt → DSPy, general → LangChain. Production stack: LangGraph + LangSmith + Letta + LlamaIndex + MCP + observability. Frontier: +reasoning (o1-style), +multimodal, +MCP, +memory, +tools.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/14-model-context-protocol, 11/16-langgraph-y-state-machines
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar framework comparison.
- Implementar decision matrix.
- Comparar LangGraph, LlamaIndex, DSPy, Letta, smolagents.
- Diagnosticar trade-offs.

## Constrúyelo

```python
def decision_matrix(requirements):
    matrix = {
        "stateful": "LangGraph",
        "rag": "LlamaIndex",
        "typescript": "Mastra",
        "memory": "Letta",
    }
    return matrix.get(requirements, "LangChain")
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: agent-frameworks
fase: 11
leccion: 17
---

1. LangGraph, LangChain, LlamaIndex.
2. DSPy, Letta, smolagents, Mastra.
3. OpenAI Agents SDK, AutoGen, CrewAI.
4. Combinaciones: state + RAG + memory.
5. Production stack.
```

## Ejercicios

1. **Frameworks**: comparar
   LangGraph vs LlamaIndex.
2. **smolagents**: implementar
   CodeAgent simple.
3. **Desafio**: production
   stack completo.

## Lecturas Recomendadas

- "LangGraph vs LangChain: When to Use Which" (LangChain, 2024)
- "smolagents: A Minimalist Agent Library" (HuggingFace, 2024)
- "LlamaIndex Agents" (LlamaIndex, 2024)
- "DSPy: Programming LLMs" (Stanford, Khattab, 2024)

---

> 📚 **Adaptación al español de la lección [Agent Framework Tradeoffs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).