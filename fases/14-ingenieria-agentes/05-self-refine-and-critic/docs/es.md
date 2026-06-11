# Self refine and critic

> Self-Refine (Madaan 2023): iterative refinement via self-critique + refine. +Quality sin RLHF, +Convergence, +Iterative, +Self-critique, -Gradient. Loop: (1) initial output, (2) critique, (3) if no issues -> stop, (4) refine, (5) repeat, (6) max iterations. CRITIC (Gou 2024): +Tool-use verification, +Verify + Correct, +External feedback, +Production. Variants: Self-Refine, CRITIC, Constitutional AI (Anthropic 2022 +Principles +Safety +Alignment), RLAIF (Google 2023 +RL +Preferences +Scalable), RLHF, DPO. Frameworks: langchain, openai, anthropic, autogen. +Production: standard 2024-25. +Use cases: agent, RAG, coding, generation, verification, safety. Decision: quality -> Self-Refine, verify -> CRITIC, safety -> Constitutional, RL -> RLAIF, production -> CRITIC o Self-Refine. Trade-offs: cada uno + specialty, Self-Refine + quality, single-shot + simple. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01, 14/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar CritiqueMemory.
- Implementar generate_critique (issues list).
- Implementar self_refine (loop).
- Implementar critic_with_tool_use (CRITIC).
- Diagnosticar Self-Refine vs CRITIC vs Constitutional AI.

## Constrúyelo

```python
def self_refine(initial_output, refine_fn, max_iterations=5, quality_threshold=0.9):
    output = initial_output
    for i in range(max_iterations):
        critique = generate_critique(output)
        if critique is None:
            return {"output": output, "iterations": i, "converged": True}
        output = refine_fn(output, critique)
    return {"output": output, "iterations": max_iterations, "converged": False}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: self-refine
fase: 14
leccion: 05
---

1. Self-critique.
2. Refine.
3. Converge.
4. CRITIC tool-use.
5. +Quality.
```

## Ejercicios

1. **Self-Refine**: implementar
   Self-Refine con LLM real.
2. **CRITIC**: probar
   CRITIC con tools.
3. **Desafio**: coding
   agent con Self-Refine.

## Lecturas recomendadas

- "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)
- "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing" (Gou et al., 2024)
- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
- "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Google, 2023)

---

> 📚 **Adaptación al español de la lección [Self Refine and Critic]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).