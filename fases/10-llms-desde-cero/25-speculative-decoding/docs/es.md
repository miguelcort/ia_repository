# Speculative decoding

> Speculative decoding (Leviathan 2023, Chen 2023): draft model chico genera k tokens, target model evalua en paralelo. Acceptance-rejection: min(1, p_target/p_draft). Si reject, sample de adjusted p' = max(0, p_target - p_draft), renormalizar. Garantiza distribución final == target distribución (proof en Leviathan 2023). Speedup 2-3x clásico. Variantes SOTA 2024-25: EAGLE-3 (self-speculative, feature-level, 2-3x), Lookahead (n-gram pool + Jacobi, 3-4x, sin training), Tree-based (EAGLE-2, 3-4x), Medusa-2 (multi-head, 2-2.5x), Draft model (mature, 2-3x). Frameworks: vLLM, SGLang, llama.cpp, TensorRT-LLM. Combos: spec + INT4 AWQ + paged = 100+ tok/sec Llama 70B A100.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/15-speculative-decoding-eagle3, 10/22-async-hogwild-inference
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar acceptance-rejection.
- Implementar adjusted distribution.
- Calcular expected speedup.
- Diagnosticar EAGLE3, Lookahead, tree-based.

## Constrúyelo

```python
def acceptance_rejection(p_target, p_draft, draft_token):
    p_t = max(p_target[draft_token], 1e-9)
    p_d = max(p_draft[draft_token], 1e-9)
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
name: prompt-speculative-decoding
fase: 10
leccion: 25
---

1. Spec decoding.
2. Acceptance min(1, p_t/p_d).
3. Adjusted distribution.
4. Distribucion == target.
5. EAGLE3, Lookahead, tree.
```

## Ejercicios

1. **Spec decoding**: implementar
   spec decoding simple.
2. **Tree**: implementar
   tree-based spec.
3. **Desafio**: Lookahead
   decoding simple.

## Lecturas recomendadas

- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
- "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)
- "Lookahead Decoding: Breaking the Sequential Dependency of LLM Inference" (Fu et al., 2024)
- "EAGLE-3: Scaling up Inference Acceleration of Large Language Models" (Li et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Speculative Decoding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).