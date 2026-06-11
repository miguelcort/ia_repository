# Eval driven agent development

> Eval-driven agent development: (1) eval suite (cases con input + expected + run + score pass rate), (2) regression detection (compare latest vs prior con threshold -5% = regression, +reliable +detectable), (3) A/B testing (compare variants +statistical), (4) CI/CD for agents, (5) continuous improvement. +Standard, +Reliable, +Production, +Detectable, +Continuous, +CI/CD, +Statistical. Components: EvalSuite (name + cases + history), add_case (name + input + expected), run (pass rate + latency + history), _matches (substring match for string), detect_regression (threshold). Variants: eval-driven, A/B testing, benchmarks, regression, statistical, custom. Frameworks: langfuse, langsmith, opik, phoenix, braintrust, custom. +Production: standard 2024-25. +Use cases: agent, eval, regression, A/B, benchmarks, production. Decision: CI/CD -> eval-driven, compare -> A/B, public -> benchmarks, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + eval-driven.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/19, 14/23
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar EvalSuite con cases + history.
- Implementar run con pass rate + latency.
- Implementar detect_regression con threshold.
- Diagnosticar eval-driven vs A/B vs benchmarks.
- Diagnosticar trade-offs.

## Constrúyelo

```python
class EvalSuite:
    def run(self, agent_fn, verbose=False):
        passed = 0
        results = []
        for case in self.cases:
            try:
                actual = agent_fn(case["input"])
                ok = self._matches(actual, case["expected"])
            except Exception as e:
                actual = f"error: {e}"
                ok = False
            if ok:
                passed += 1
        return {"score": passed / len(self.cases), "passed": passed, "total": len(self.cases)}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: eval-driven
fase: 14
leccion: 30
---

1. Eval suite.
2. Regression detection.
3. A/B testing.
4. CI/CD.
5. +Production.
```

## Ejercicios

1. **Eval suite**: implementar
   eval suite custom.
2. **Regression**: probar
   regression detection.
3. **Desafio**: CI/CD
   pipeline con evals.

## Lecturas recomendadas

- "Eval-Driven Development for LLMs" (Anthropic, 2024)
- "Braintrust: AI Eval Platform" (Braintrust, 2024)
- "Langfuse Evaluations" (Langfuse, 2024)
- "CI/CD for AI Agents" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Eval Driven Agent Development]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).