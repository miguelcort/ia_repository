# 52 — Experiment runner

> Experiment runner: ejecutar code experiments. Sandbox (Docker, Modal, E2B), parameter sweep, results logging, reproducibility. Frameworks: MLflow, W&B, Neptune, Aim.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/19, Fase 17
**Tiempo estimado:** ~30 minutos

## Objetivos

- Sandboxed execution.
- Param sweeps.
- Logging (W&B, MLflow).
- Reproducibility.

## Constrúyelo

```python
import mlflow
import subprocess


def run_experiment(code, params, output_dir):
    """Run experiment con MLflow tracking."""
    mlflow.start_run()
    mlflow.log_params(params)
    # Execute in sandbox
    result = subprocess.run(["python", code], capture_output=True)
    mlflow.log_artifact(output_dir)
    mlflow.end_run()
    return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-exp-runner
fase: 19
leccion: 52
---

1. Sandbox (Docker, E2B).
2. Param sweeps.
3. MLflow / W&B.
4. Reproducibility.
```

## Ejercicios

1. **Run**: 10 params.
2. **Sweep**: grid.
3. **Desafío**: 1000
   experiments async.

## Lecturas recomendadas

- "MLflow" (Databricks 2018)
- "Weights & Biases" (2017)
- "Modal" (2023)
- "E2B" (2024)



## Detalles avanzados

Esta lección cubre los trade-offs críticos de
producción. Considera scaling: en pre-training el
factor dominante es cómputo disponible; en inference
es latencia y costo. Frameworks standard: PyTorch
(HF Transformers, TRL, vLLM), JAX (Flax, Optax).
Optimizaciones: FlashAttention-2, paged attention,
KV cache compression, speculative decoding, MoE.

Eval riguroso: statistical significance testing
sobre múltiples seeds, held-out test sets sin
contamination, y edge cases del domain. Métricas:
BLEU/ROUGE para text generation, exact match/F1
para QA, pass@k para code, human preference para
chat.

Trampas comunes: data leakage entre train/test,
overfitting al validation set, eval con prompts
fuera de distribución, ignore de tail latency en
serving, cost runaway en production.

Tools clave: Weights & Biases o MLflow para
tracking, Langfuse para LLM observability, Hydra
para config, Ray para distributed execution, vLLM
para serving LLM. Conoce al menos uno a fondo antes
de producción.

---

> 📚 **Adaptación al español** de la lección
> "[52-experiment-runner]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
