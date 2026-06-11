# Prompt engineering

> Prompt engineering: zero-shot, few-shot (2-5 examples), system prompts. Chain-of-Thought (CoT, Wei 2022): "razona paso a paso" → +5-30% en math, logic. ReAct (Yao 2022): Thought + Action + Observation, con tools, base de agentic LLMs. Self-consistency (Wang 2022): muestreo múltiple + majority vote. Tree of Thoughts (ToT, Yao 2023): search tree. Frameworks: LangChain, LlamaIndex, DSPy (Stanford, automatic prompt optimization), promptfoo, LangSmith, LangFuse. SOTA 2024-25: context engineering, agentic prompting, reasoning models (o1, R1), function calling, MCP, multi-modal.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/06-instruction-tuning-sft
**Tiempo estimado:** ~30 minutos

## Objetivos

- Construir prompts con system, context, examples.
- Implementar CoT y ReAct templates.
- Estimar tokens de prompt.
- Diagnosticar frameworks y eval.

## Constrúyelo

```python
def build_prompt(instruction, context=None, examples=None, system=None):
    parts = []
    if system: parts.append(f"### System:\n{system}\n")
    if context: parts.append(f"### Context:\n{context}\n")
    if examples:
        for ex in examples:
            parts.append(f"### Example:\nInput: {ex[0]}\nOutput: {ex[1]}\n")
    parts.append(f"### Instruction:\n{instruction}\n\n### Response:\n")
    return "\n".join(parts)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-engineering
fase: 11
leccion: 01
---

1. Zero-shot, few-shot, system.
2. CoT: razonar paso a paso.
3. ReAct: Thought/Action/Observation.
4. +5-30% quality.
5. DSPy, promptfoo, LM-as-judge.
```

## Ejercicios

1. **CoT**: implementar few-shot
   CoT en math problem.
2. **ReAct**: implementar agente
   ReAct con tool (search).
3. **Desafio**: DSPy pipeline
   automatic prompt opt.

## Lecturas recomendadas

- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
- "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (Khattab et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Prompt Engineering]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).