# 72 — Code execution metric

> Code exec metric: ejecutar generated code con tests, verificar output. HumanEval, MBPP, SWE-bench, LiveCodeBench. Pass@k estimator. Sandbox para safe execution.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/71
**Tiempo estimado:** ~20 minutos

## Objetivos

- Pass@k.
- Sandbox execution.
- Test runner.
- HumanEval, SWE-bench.

## Constrúyelo

```python
from math import comb


def pass_at_k(n, c, k):
    """1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def execute_code_test(code, test_cases, timeout=10):
    """Execute code with test cases."""
    results = []
    for test in test_cases:
        try:
            ns = {}
            exec(code, ns)
            result = ns["solve"](*test["args"])
            results.append(result == test["expected"])
        except Exception:
            results.append(False)
    return sum(results) / len(test_cases)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-code-exec-metric
fase: 19
leccion: 72
---

1. Pass@k.
2. Sandbox exec.
3. Test runner.
4. HumanEval.
```

## Ejercicios

1. **Pass@1**: 10 tasks.
2. **Pass@10**: 50
   samples.
3. **Desafío**: SWE-bench
   Verified.

## Lecturas recomendadas

- "HumanEval" (Chen 2021)
- "SWE-bench" (Jimenez 2024)
- "LiveCodeBench" (2024)



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
> "[72-code-exec-metric]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
