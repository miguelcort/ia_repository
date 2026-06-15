# 57 — End-to-end research demo

> End-to-end research demo: agent toma research question, genera hipótesis, corre experimentos, escribe paper. Pipeline integration de lecciones 50-56. Frameworks: AI co-scientist (Google), STORM, deep research.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/50-56
**Tiempo estimado:** 30 horas

## Objetivos

- Pipeline integration.
- Hypothesis → exp → paper.
- Quality eval.
- Multi-iteration.

## Constrúyelo

```python
class ResearchPipeline:
    def __init__(self, llm, sandbox, evaluator):
        self.llm = llm
        self.sandbox = sandbox
        self.evaluator = evaluator

    def run(self, research_question, max_iters=5):
        # 1. Hypothesis
        hypotheses = self.generate_hypotheses(research_question)
        # 2. Iterative refinement
        for i in range(max_iters):
            # Run experiments
            results = []
            for h in hypotheses:
                code = self.generate_code(h)
                result = self.sandbox.run(code)
                results.append(result)
            # Evaluate
            best = self.evaluator.rank(results)
            # Refine
            hypotheses = self.refine(hypotheses, best)
        # Write paper
        return self.write_paper(hypotheses, results)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-research-pipeline
fase: 19
leccion: 57
---

1. Question.
2. Hypotheses.
3. Experiments.
4. Eval.
5. Paper.
```

## Ejercicios

1. **Mini-demo**: ML
   research question.
2. **Iterate**: 3
   iterations.
3. **Desafío**: full
   paper draft.

## Lecturas recomendadas

- "AI Co-scientist" (Google 2024)
- "STORM" (Shao 2024)
- "DeepResearch" (OpenAI 2025)

---

> 📚 **Adaptación al español** de la lección
> "[57-end-to-end-research-demo]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
