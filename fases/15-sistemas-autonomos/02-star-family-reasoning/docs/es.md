# Star family reasoning

> Star family reasoning: (1) STaR (Zelikman 2022 Self-Taught Reasoner, generate rationale + answer, filter correct, fine-tune on filtered, +reasoning +iterative +standard +training), (2) Quiet-STaR (Zelikman 2024, parallel reasoning tokens, pick longest, +efficient +inference), (3) V-STaR (verify + self-consistency +majority vote), (4) ReST (Reinforced Self-Training, filter correct + train, +reinforced +filter +augment). +Reasoning, +Self-improving, +Iterative, +Efficient, +Standard, +Reliable, +Production. Variants: STaR seminal, Quiet-STaR, V-STaR, ReST, custom. Frameworks: openai, anthropic, huggingface, custom, trl, peft. +Production: standard 2024-25. +Use cases: reasoning, math, code, training, inference, fine-tuning. Decision: training -> STaR, inference -> Quiet-STaR, verify -> V-STaR, filter -> ReST, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01, 11/15
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar star_generate_rationale + star_filter_correct.
- Implementar quiet_star_reasoning (parallel passes).
- Implementar v_star_self_consistency (majority vote).
- Implementar rest_train (filter + train).
- Diagnosticar STaR vs Quiet-STaR vs V-STaR vs ReST.

## Constrúyelo

```python
def star_generate_rationale(question, base_answer_fn):
    rationale = base_answer_fn(f"Explain step by step: {question}")
    answer = base_answer_fn(f"Given: {rationale}. Answer: {question}")
    return {"rationale": rationale, "answer": answer}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: star-reasoning
fase: 15
leccion: 02
---

1. STaR.
2. Quiet-STaR.
3. V-STaR.
4. ReST.
5. +Self-improving.
```

## Ejercicios

1. **STaR**: implementar
   STaR con fine-tuning.
2. **Quiet-STaR**: probar
   parallel reasoning.
3. **Desafio**: full
   star pipeline.

## Lecturas recomendadas

- "STaR: Self-Taught Reasoner" (Zelikman et al., 2022)
- "Quiet-STaR" (Zelikman et al., 2024)
- "ReST: Reinforced Self-Training" (Gulcehre et al., 2023)
- "V-STaR: Verify + Self-Consistency" (Hosseini et al., 2024)

---

> 📚 **Adaptación al español de la lección [Star Family Reasoning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).