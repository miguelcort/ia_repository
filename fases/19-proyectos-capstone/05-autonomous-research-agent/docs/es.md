# 05 — Autonomous research agent

> Autonomous research agent: lee papers, genera hipótesis, corre experimentos, escribe drafts. Sistemas como DeepResearch (OpenAI), ScholarQA, STORM. Multi-step, requiere planning, tool use, y critic loop.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 11, Fase 14 (agents), Fase 13 (tools)
**Tiempo estimado:** 30 horas

## Objetivos

- Implementar agent con literature retrieval.
- Hipótesis generator + experiment runner.
- Critic loop para verificar resultados.
- Evaluar quality de research output.

## El problema

Research agents (DeepResearch OpenAI, STORM Stanford,
ScholarGPT) automatizan partes de literature review
y discovery. Pipeline: (1) Hypothesis generator
(LLM). (2) Literature retrieval (semantic scholar,
arxiv, OpenAlex). (3) Experiment runner (código).
(4) Result evaluator (LLM judge). (5) Paper writer
(LLM con citations). (6) Critic loop: el LLM
identifica weaknesses y propone revisions. Iterar
hasta quality threshold.

## Constrúyelo

```python
class ResearchAgent:
    def __init__(self, llm, retriever, runner):
        self.llm = llm
        self.retriever = retriever
        self.runner = runner
        self.hypotheses = []
        self.experiments = []

    def run(self, question, max_iters=5):
        for i in range(max_iters):
            hyp = self.generate_hypothesis(question)
            self.hypotheses.append(hyp)
            papers = self.retriever.search(hyp)
            exp = self.design_experiment(hyp, papers)
            result = self.runner.run(exp)
            self.experiments.append(result)
            critique = self.critique(result)
            if self.is_good(critique):
                return self.write_paper(result, papers)
        return self.write_paper(self.experiments[-1], papers)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-research-agent
fase: 19
leccion: 05
---

1. Hypothesis generation.
2. Literature retrieval.
3. Experiment design.
4. Result evaluation.
5. Iterative critic loop.
```

## Ejercicios

1. **Agent setup**: integrar
   semantic scholar API.
2. **Critic loop**: implementar
   self-correction.
3. **Desafío**: end-to-end
   ML research task.

## Lecturas recomendadas

- "STORM" (Shao 2024, Stanford)
- "DeepResearch" (OpenAI 2025)
- "ScholarQA" (Allen AI 2024)
- "AutoML-Agent" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[05-autonomous-research-agent]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
