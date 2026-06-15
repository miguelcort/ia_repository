# 41 — Eval pipeline

> Eval pipeline: lm-eval-harness (100+ benchmarks, EleutherAI), HELM (holistic, Stanford), AlpacaEval, custom. Components: task loader, model runner, scoring, aggregation. CI/CD integration.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/37
**Tiempo estimado:** ~30 minutos

## Objetivos

- Run lm-eval-harness.
- Custom benchmark.
- Scoring.
- CI/CD integration.

## Constrúyelo

```python
import lm_eval
from lm_eval.models.huggingface import HFLM


def run_lm_eval(model_id, tasks, batch_size=4):
    """Run lm-evaluation-harness."""
    model = HFLM(pretrained=model_id, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=model, tasks=tasks, batch_size=batch_size)
    return results


def custom_benchmark(items, model, scoring_fn):
    """Custom eval: items = [(input, expected)], scoring."""
    scores = []
    for inp, expected in items:
        output = model(inp)
        scores.append(scoring_fn(output, expected))
    return {"mean": sum(scores) / max(len(scores), 1)}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-eval-pipeline
fase: 19
leccion: 41
---

1. lm-eval-harness.
2. Custom tasks.
3. Scoring.
4. CI integration.
5. Reports.
```

## Ejercicios

1. **MMLU**: 5-shot
   en Llama 3 8B.
2. **Custom**: 100 q&a
   domain-specific.
3. **Desafío**: leaderboard
   propio.

## Detalles

lm-eval-harness (Gao 2021, EleutherAI): 200+ tasks.
MMLU, HellaSwag, ARC, GSM8K, HumanEval, TruthfulQA,
MMLU-Pro, GPQA, BIG-Bench, etc. CLI: `lm_eval
--model hf --model_args pretrained=... --tasks mmlu
--num_fewshot 5`. Python API via HFLM, vLLM, etc.

HELM (Liang 2022, Stanford): holistic eval con
multi-metric (accuracy, robustness, bias, fairness).
Más lento pero comprehensivo. v1 con 30+ scenarios.

AlpacaEval (Li 2023, Stanford): LLM-as-judge, win
rate vs reference (Davinci-003 o GPT-4). MT-Bench
multi-turn. Chatbot Arena: human preference Elo.

Custom tasks: YAML format. Define `dataset_path`,
`output_type` (multiple_choice, loglikelihood,
generate), `doc_to_text`, `doc_to_target`, `metric_list`.

CI/CD: GitHub Actions corre eval en cada PR. Threshold
gates (e.g. MMLU > 60%, GSM8K > 40%). Alertas en
regression.

Leaderboards: OpenLLM (deprecated), Artificial
Analysis, LMSYS Arena, Stanford CRFM. Publican
resultados de frontier models.

Hoy: lm-eval-harness + LMSYS Arena + internal evals
es el standard. Custom evals para domain-specific.

## Lecturas recomendadas

- "lm-eval-harness" (Gao 2021)
- "HELM" (Liang 2022)
- "AlpacaEval" (Li 2023)
- "LiveBench" (2024)
- "MMLU-Pro" (Wang 2024)

---

> 📚 **Adaptación al español** de la lección
> "[41-eval-pipeline]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
