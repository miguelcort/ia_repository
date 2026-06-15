# 82 — Jailbreak taxonomy

> Jailbreak taxonomy (OWASP LLM Top 10, MITRE ATLAS): categorías de ataques. Prompt injection, jailbreak, model theft, data poisoning. Defense patterns por categoría.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 18/12-15
**Tiempo estimado:** ~20 minutos

## Objetivos

- Attack taxonomy.
- Defense mapping.
- Eval por categoría.
- OWASP / MITRE.

## Constrúyelo

```python
JAILBREAK_TAXONOMY = {
    "prompt_injection": {
        "direct": ["ignore previous", "system override"],
        "indirect": ["via tool output", "via RAG"],
        "defenses": ["tagging", "spotlighting", "STR"],
    },
    "jailbreak": {
        "roleplay": ["DAN", "evil twin"],
        "many_shot": ["256-shot dialogues"],
        "ascii": ["art-based"],
        "defenses": ["LlamaGuard", "constitutional"],
    },
    "data_exfiltration": {
        "training_data": ["MI attacks", "membership inference"],
        "system_prompt": ["leak via prompt"],
        "defenses": ["DP", "differential privacy"],
    },
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
name: prompt-jailbreak-taxonomy
fase: 19
leccion: 82
---

1. Taxonomy.
2. OWASP LLM Top 10.
3. MITRE ATLAS.
4. Defenses.
```

## Ejercicios

1. **Categorize**: 50
   attacks.
2. **Defense mapping**.
3. **Desafío**: eval
   suite por cat.

## Lecturas recomendadas

- "OWASP LLM Top 10" (2024)
- "MITRE ATLAS" (2024)
- "HarmBench" (Mazeika 2024)



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
> "[82-jailbreak-taxonomy]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
