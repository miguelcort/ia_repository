# 70 — Task spec format

> Task spec: standardized format for eval tasks. Fields: id, input, expected, scoring_fn, timeout, dependencies. Frameworks: SWE-bench, AgentBench, custom YAML/JSON.

**Tipo:** Construir
**Lenguajes:** Python, YAML
**Prerrequisitos:** Fase 19/27
**Tiempo estimado:** ~20 minutos

## Objetivos

- YAML task format.
- Schema validation.
- Loader.
- Run loop integration.

## Constrúyelo

```python
import yaml
from jsonschema import validate


TASK_SCHEMA = {
    "type": "object",
    "required": ["id", "input", "expected", "scoring"],
    "properties": {
        "id": {"type": "string"},
        "input": {"type": "string"},
        "expected": {"type": "string"},
        "scoring": {"enum": ["exact", "fuzzy", "exec", "llm"]},
        "timeout": {"type": "number"},
    },
}


def load_task_spec(path):
    with open(path) as f:
        task = yaml.safe_load(f)
    validate(task, TASK_SCHEMA)
    return task
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-task-spec
fase: 19
leccion: 70
---

1. YAML format.
2. JSON Schema validation.
3. Scoring functions.
4. Run loop.
```

## Ejercicios

1. **10 tasks** YAML.
2. **Validate** schema.
3. **Desafío**: 100 tasks
   automated.

## Lecturas recomendadas

- "SWE-bench" (Jimenez 2024)
- "AgentBench" (Liu 2023)
- "JSON Schema" (2024)



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
> "[70-task-spec-format]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
