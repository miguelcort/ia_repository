# Speculative decoding EAGLE3

> EAGLE3 (Li 2024, Microsoft): self-speculative decoding, opera en features (no tokens), multi-layer draft en single forward, distillation lightweight (1% training overhead), 2-3x speedup en inference, no draft model separado. Acceptance: min(1, p_target/p_draft), típico 0.5-0.8. Vs draft model: comparable speedup, sin overhead de modelo extra. Vs Medusa: token-level multi-head, 2-2.5x. Hoy: EAGLE3 es gold standard para LLMs SOTA (Llama 3, Qwen 2.5, Mixtral). Frameworks: EAGLE-3, vLLM EAGLE plugin, SGLang, llama.cpp spec. Combo: EAGLE3 + INT4 AWQ + vLLM = Llama 3 70B 50-100+ tok/sec en A100.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/12-optimizacion-de-inferencia
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar acceptance criterion.
- Calcular EAGLE3 speedup estimado.
- Comparar EAGLE3 vs draft model vs Medusa.
- Diagnosticar acceptance rate.

## Constrúyelo

```python
def speculative_acceptance(target_probs, draft_probs, draft_token):
    p_t = max(target_probs[draft_token], 1e-9)
    p_d = max(draft_probs[draft_token], 1e-9)
    return min(1.0, p_t / p_d)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-eagle3
fase: 10
leccion: 15
---

1. EAGLE3: feature-level.
2. Mismo model, distillation.
3. 2-3x speedup.
4. Acceptance 0.5-0.8.
5. vLLM EAGLE plugin.
```

## Ejercicios

1. **EAGLE3**: implementar
   EAGLE3-style en Llama 3 8B.
2. **Medusa**: comparar EAGLE3
   vs Medusa.
3. **Desafio**: tree-based
   spec decoding.

## Lecturas recomendadas

- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al., 2024)
- "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (Li et al., 2024)
- "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" (Li et al., 2024)
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Speculative Decoding EAGLE3]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).