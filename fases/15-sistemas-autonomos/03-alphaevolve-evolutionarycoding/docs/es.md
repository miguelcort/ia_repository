# AlphaEvolve evolutionary coding

> AlphaEvolve (Google DeepMind 2025): evolutionary coding agent. Code mutation + selection on benchmarks. +SOTA on math + code. +Production, +Evolutionary, +Convergence, +Standard, +Reliable. Mutations: (1) replace (char -> char, +Simple), (2) insert (char position, +Length), (3) delete (remove char, -Length), (4) swap (exchange, +Order), (5) crossover (combine 2, +Diversity). Population: random + initial (+Diverse) -> Mutate (+Variations) -> Evaluate (benchmark score +Standard) -> Select top (+Elitism) -> Repeat N generations -> +Convergence. Variants: AlphaEvolve (Google DeepMind 2025 +SOTA +general +evolutionary +math +code), AlphaCode (DeepMind 2022 +competitive +code +standard), FunSearch (DeepMind 2023 +math +evolutionary +discovery), custom, OpenAI Codex, Anthropic Claude Code, Codeium. Frameworks: custom, deepmind, openai, anthropic, codeium. +Production: standard 2024-25. +Use cases: coding, math, optimization, algorithms, evolution. Decision: SOTA -> AlphaEvolve, code -> AlphaCode, math -> FunSearch, simple -> custom, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + evolutionary.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/11, 15/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mutate_code (replace/insert/delete).
- Implementar evaluate_code con eval_fn.
- Implementar alphaevolve con population + select.
- Diagnosticar mutations types.
- Diagnosticar AlphaEvolve vs AlphaCode vs FunSearch.

## Constrúyelo

```python
def mutate_code(code, n_mutations=1):
    code = list(code)
    for _ in range(n_mutations):
        op = random.choice(["replace", "insert", "delete"])
        if op == "replace":
            code[random.randint(0, len(code) - 1)] = random.choice("abc...")
        elif op == "insert":
            code.insert(random.randint(0, len(code)), random.choice("abc..."))
        elif op == "delete" and code:
            del code[random.randint(0, len(code) - 1)]
    return "".join(code)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: alphaevolve
fase: 15
leccion: 03
---

1. Evolutionary coding.
2. Mutation + selection.
3. Population + generations.
4. +SOTA math + code.
5. +Production.
```

## Ejercicios

1. **AlphaEvolve**: implementar
   custom evolutionary coding.
2. **Mutations**: probar
   diferentes mutations.
3. **Desafio**: full
   evolutionary pipeline.

## Lecturas recomendadas

- "AlphaEvolve: Evolutionary Coding Agent" (Google DeepMind, 2025)
- "AlphaCode: Competitive Code Generation" (Li et al., DeepMind, 2022)
- "FunSearch: Mathematical Discovery via LLMs" (Romera-Paredes et al., DeepMind, 2023)

---

> 📚 **Adaptación al español de la lección [AlphaEvolve Evolutionary Coding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).