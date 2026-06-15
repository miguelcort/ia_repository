# 49 — LM eval harness

> LM eval harness (EleutherAI, Gao 2021): 100+ benchmarks, unified interface. MMLU, HellaSwag, ARC, GSM8K, HumanEval, TruthfulQA, etc. CLI + Python API. Custom tasks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/37, 19/41
**Tiempo estimado:** ~25 minutos

## Objetivos

- Install lm-eval-harness.
- Run MMLU suite.
- Custom task.
- Report.

## Constrúyelo

```python
import lm_eval
from lm_eval.models.huggingface import HFLM


def run_mmlu(model_id, n_shot=5):
    """MMLU 5-shot eval."""
    model = HFLM(pretrained=model_id)
    results = lm_eval.simple_evaluate(
        model=model, tasks=["mmlu"], num_fewshot=n_shot)
    return results["results"]["mmlu"]


def custom_task_yaml(path):
    """Write custom task YAML."""
    return {
        "task": "my_task",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": path},
        "output_type": "multiple_choice",
        "doc_to_text": "{{question}}",
        "doc_to_target": "answer",
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-lm-eval
fase: 19
leccion: 49
---

1. lm-eval-harness.
2. MMLU, GSM8K, etc.
3. Custom task.
4. Reports.
```

## Ejercicios

1. **MMLU**: Llama 3 8B
   5-shot.
2. **Custom**: 100 q&a
   domain.
3. **Desafío**: leaderboard
   propio.

## Detalles

lm-evaluation-harness (Gao 2021, EleutherAI): framework
estándar para evaluar LLMs. 200+ tasks. CLI: `lm_eval
--model hf --model_args pretrained=meta-llama/
Llama-3-8B --tasks mmlu,hellaswag,gsm8k --num_fewshot 5
--batch_size 4 --output_path results/`.

Tasks: knowledge (MMLU, ARC), reasoning (HellaSwag,
WinoGrande), math (GSM8K, MATH), code (HumanEval, MBPP),
truthfulness (TruthfulQA, BBQ), multilingual (XCOPA,
XNLI), long-context (QuALITY, Needle).

Output types: (1) multiple_choice: loglikelihood-based.
(2) generate: free-form generation. (3) loglikelihood:
perplexity over options.

Custom task YAML:
```yaml
task: my_task
dataset_path: json
dataset_kwargs:
  data_files: data.jsonl
output_type: multiple_choice
doc_to_text: "{{question}}"
doc_to_target: "answer"
metric_list:
  - metric: acc
```

CI/CD: GitHub Action corre eval en PR. Threshold
regression detection. Slack/email alert.

Hoje: lm-eval-harness + HELM + LMSYS Arena + custom
evals. OpenLLM Leaderboard v2 usa subset (IFEval,
BBH, MATH, GPQA, MMLU-Pro, MUSR).

## Lecturas recomendadas

- "lm-evaluation-harness"
  (Gao 2021, EleutherAI)
- "MMLU" (Hendrycks 2020)
- "OpenLLM Leaderboard"
  (Hugging Face 2024)
- "HELM" (Liang 2022)

---

> 📚 **Adaptación al español** de la lección
> "[49-lm-eval-harness]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
