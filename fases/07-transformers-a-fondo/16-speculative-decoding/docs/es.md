# Speculative decoding

> Speculative decoding: draft model chico genera k tokens, target model grande valida en paralelo. Acceptance-rejection: min(1, p_target/p_draft). Speedup 2-3x con draft 10x más chico, gamma 4-8. Output distribution = target distribution (exacta). Self-speculative: Medusa (k heads), EAGLE. Frameworks: vLLM, TGI, SGLang, llama.cpp.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/07-gpt-causal-language-modeling
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar acceptance-rejection criterion.
- Implementar distribution adjustment.
- Construir speculative decoder con draft + target.
- Calcular speedup esperado.

## Constrúyelo

```python
def acceptance_rejection(p_draft, p_target, token, eps=1e-9):
    p_d_t = max(p_draft[token], eps)
    p_t_t = max(p_target[token], eps)
    return min(1.0, p_t_t / p_d_t)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-speculative-decoding
fase: 07
leccion: 16
---

1. Draft chico, target grande.
2. Accept: min(1, p_target/p_draft).
3. Speedup 2-3x, gamma 4-8.
4. Medusa, EAGLE: self-speculative.
5. vLLM, TGI, SGLang en production.
```

## Ejercicios

1. **Benchmark**: medir speedup de speculative
   decoding en small LM.
2. **Medusa**: agregar k heads al modelo y
   medir aceptacion.
3. **Desafio**: implementar tree-based
   speculative decoding.

## Lecturas recomendadas

- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
- "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)
- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (Cai et al., 2024)
- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Speculative Decoding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).