# Anthropic workflow patterns

> Anthropic workflow patterns (Building Effective Agents, 2024): (1) prompt chaining (sequential, each step -> next), (2) routing (classify + dispatch to handler), (3) parallelization (concurrent LLM calls), (4) orchestrator-workers (decompose + execute parallel + synthesize, +hierarchical +modular +parallel +efficient +reliable +scalable), (5) evaluator-optimizer (generate + evaluate score + if good stop + optimize refine + repeat max iterations, +iterative +quality +refined +reliable). +Structured, +Composable, +Production, +Reliable, +Scalable, +Modular. Variants: chaining, routing, parallel, orchestrator, eval-opt, hierarchical, map-reduce, Self-Refine, CRITIC, Reflexion. Frameworks: anthropic, langchain, openai, langgraph, autogen, crewai, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, automation, document, code, generation. Decision: sequential -> chaining, classify -> routing, fast -> parallel, hierarchical -> orchestrator, quality -> eval-opt, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + patterns.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar prompt_chaining.
- Implementar routing.
- Implementar parallelization.
- Implementar orchestrator_workers.
- Implementar evaluator_optimizer.
- Diagnosticar patterns trade-offs.

## Constrúyelo

```python
def orchestrator_workers(query, orchestrator_fn, worker_fn, n_workers=3):
    subtasks = orchestrator_fn(query, n=n_workers)
    results = [worker_fn(sub) for sub in subtasks]
    return orchestrator_fn(query, results=results, mode="synthesize")
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: anthropic-patterns
fase: 14
leccion: 12
---

1. Chaining.
2. Routing.
3. Parallel.
4. Orchestrator.
5. Eval-opt.
```

## Ejercicios

1. **Anthropic patterns**: usar
   Building Effective Agents.
2. **Orchestrator**: implementar
   orchestrator custom.
3. **Desafio**: production
   workflow combinando patterns.

## Lecturas recomendadas

- "Building Effective Agents" (Anthropic, 2024)
- "LangChain Expression Language" (LangChain, 2024)
- "LangGraph Workflows" (LangChain, 2024)
- "Anthropic Claude Best Practices" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Anthropic Workflow Patterns]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).