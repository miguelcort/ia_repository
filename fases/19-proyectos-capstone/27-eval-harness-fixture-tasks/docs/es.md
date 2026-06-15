# 27 — Eval harness con fixture tasks

> Eval harness: framework para correr tasks (fixtures) contra LLM agent. Componentes: task spec, expected output, scoring function, run loop. Frameworks: SWE-bench, LiveCodeBench, lm-eval-harness.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 11, Fase 19/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Task spec format (JSON/YAML).
- Fixture loading.
- Scoring function.
- Run + report loop.

## El problema

Eval harness: dataset de tasks con (input, expected
output, scoring function). Run loop: para cada task,
ejecutar agent, comparar output vs expected, calcular
score. Frameworks: SWE-bench (real GitHub issues,
2K+), LiveCodeBench (contest problems, 400+),
lm-eval-harness (100+ benchmarks), AgentBench
(multi-step). Componentes: (1) Task spec (input,
expected, scoring). (2) Fixtures loader. (3) Run loop.
(4) Metrics (pass@k, time). (5) Report.

## Constrúyelo

```python
import yaml
import json


def load_tasks(path):
    """Load eval tasks from YAML/JSON."""
    if path.endswith(".yaml"):
        return yaml.safe_load(open(path))
    return json.load(open(path))


def run_eval(agent, tasks):
    results = []
    for task in tasks:
        output = agent.run(task["input"])
        score = task["scoring"](output, task["expected"])
        results.append({"task_id": task["id"],
                       "score": score,
                       "output": output})
    return {"mean_score": sum(r["score"] for r in results) / len(results),
            "results": results}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-eval-harness
fase: 19
leccion: 27
---

1. Task spec (YAML/JSON).
2. Fixture loading.
3. Scoring function.
4. Run loop.
5. Metrics + report.
```

## Ejercicios

1. **Tasks**: 10 fixture
   tasks.
2. **Run**: medir pass@k.
3. **Desafío**: SWE-bench
   Lite integration.

## Lecturas recomendadas

- "SWE-bench" (Jimenez 2024)
- "LiveCodeBench" (2024)
- "AgentBench" (Liu 2023)
- "lm-eval-harness" (EleutherAI)

---

> 📚 **Adaptación al español** de la lección
> "[27-eval-harness-fixture-tasks]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
