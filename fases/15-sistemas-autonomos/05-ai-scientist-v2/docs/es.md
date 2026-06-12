# AI scientist v2

> AI Scientist v2 (Sakana 2025): autonomous research agent. Generates ideas + experiments + papers. End-to-end research cycle: (1) generate_idea (title + hypothesis + methodology), (2) run_experiment (metric + significant), (3) write_paper (abstract + intro + method + results + conclusion), (4) +iterative +self-improving +autonomous +end-to-end +open-ended. +Autonomous, +Research, +Open-ended, +Self-improving, +Iterative, +Independent, +Quality, +New ideas, +Discovery, +Papers, +Published output, +Reproduction, +Standard, +Reliable, +Scalable. Variants: AI Scientist v2 seminal, Research agent, custom, openai deep research, anthropic research. Frameworks: sakana, openai, anthropic, custom, langgraph. +Production: standard 2024-25. +Use cases: research, automation, ML, science, papers. Decision: research -> AI Scientist v2, self-improving -> DGM, custom -> custom, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + research.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ResearchIdea class.
- Implementar AIScientistV2 con cycle.
- Diagnosticar research cycle.
- Diagnosticar AI Scientist v2 vs DGM vs custom.

## Constrúyelo

```python
class AIScientistV2:
    def research_cycle(self, llm_fn, topic, n_iterations=3):
        results = []
        for i in range(n_iterations):
            self.iteration += 1
            idea = self.generate_idea(llm_fn, topic)
            experiment = self.run_experiment(idea)
            paper = self.write_paper(idea, experiment)
            results.append({"idea": idea, "experiment": experiment, "paper": paper})
        return results
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: ai-scientist
fase: 15
leccion: 05
---

1. Autonomous research.
2. Generate + experiment + paper.
3. Research cycle.
4. +Self-improving.
5. +Open-ended.
```

## Ejercicios

1. **AI Scientist**: implementar
   research agent custom.
2. **Papers**: probar
   paper generation.
3. **Desafio**: full
   research pipeline.

## Lecturas recomendadas

- "AI Scientist v2: Autonomous Research" (Sakana AI, 2025)
- "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (Lu et al., Sakana, 2024)
- "Sakana AI Scientist" (Sakana, 2024)
- "Automated Research with LLMs" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [AI Scientist v2]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).