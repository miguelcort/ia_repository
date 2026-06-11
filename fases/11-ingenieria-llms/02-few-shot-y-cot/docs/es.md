# Few-shot y Chain-of-Thought

> Few-shot prompting: 2-5 examples representativos, +5-15% quality. Chain-of-Thought (CoT, Wei 2022): "razona paso a paso" → +20-30% en GSM8K, MATH, logic. Zero-shot CoT (Kojima 2022): solo magic phrase. Self-consistency (Wang 2022): sample N=10-40 independent paths, majority vote, +5-10% sobre single-path CoT. Tree of Thoughts (ToT, Yao 2023): search tree con BFS/DFS, +20-50% en planning. Graph of Thoughts (GoT): DAG. Reasoning models 2024-25: o1, o3 (OpenAI, test-time compute, search + verifier), DeepSeek-R1 (GRPO, +reasoning), o3, o1-pro, QwQ, Llama 4 reasoning. Process reward models (PRM, step-level rewards, o1-style). Hoy: o1 + R1 son SOTA reasoning, frontier es o3-style search + PRM.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/01-prompt-engineering
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar few-shot prompt builder.
- Implementar CoT y self-consistency.
- Diagnosticar ToT y variantes.
- Comparar reasoning models.

## Constrúyelo

```python
def few_shot_prompt(instruction, examples, format_str="Q: {q}\nA: {a}"):
    parts = []
    for ex in examples:
        parts.append(format_str.format(q=ex[0], a=ex[1]))
    parts.append(format_str.format(q=instruction, a=""))
    return "\n\n".join(parts)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: few-shot-cot
fase: 11
leccion: 02
---

1. Few-shot 2-5 examples.
2. CoT: paso a paso.
3. Self-consistency N=10-40.
4. ToT search.
5. o1, R1 reasoning.
```

## Ejercicios

1. **CoT**: few-shot CoT en
   math problem.
2. **Self-consistency**: implementar
   majority vote.
3. **Desafio**: ToT para
   24-game.

## Lecturas recomendadas

- "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
- "Self-Consistency Improves Chain of Thought" (Wang et al., 2022)
- "Tree of Thoughts" (Yao et al., 2023)
- "DeepSeek-R1: Incentivizing Reasoning Capability" (Guo et al., 2025)

---

> 📚 **Adaptación al español** de la lección "[Few-Shot CoT]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).