# 75 — End-to-end eval runner

> End-to-end eval runner: orchestration de benchmarks, models, datasets, scoring. CI/CD integration, results storage, comparisons. Frameworks: lm-eval-harness, deep_eval, custom.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/27, 19/70-74
**Tiempo estimado:** ~25 minutos

## Objetivos

- Multi-model runner.
- Multi-benchmark orchestration.
- Results DB.
- Report generation.

## Constrúyelo

```python
def run_eval_suite(models, benchmarks, output_dir):
    """Run all models x all benchmarks."""
    results = {}
    for model_id in models:
        results[model_id] = {}
        for bench in benchmarks:
            scores = run_benchmark(model_id, bench)
            results[model_id][bench] = scores
    save_results(results, output_dir)
    generate_report(results)
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
name: prompt-e2e-eval
fase: 19
leccion: 75
---

1. Multi-model.
2. Multi-benchmark.
3. Results DB.
4. Report.
```

## Ejercicios

1. **3 models** x 5
   benchmarks.
2. **Compare** baselines.
3. **Desafío**: CI/CD
   integration.

## Lecturas recomendadas

- "lm-eval-harness" (2021)
- "deep_eval" (2024)
- "Big-Bench" (2022)



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
> "[75-end-to-end-eval-runner]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
