# EAGLE3 speculative decoding

> EAGLE3: (1) Draft (small+fast), (2) Target (big+verify), (3) Parallel (verify+multiple), (4) Speedup (2-3x+latency reduce), (5) No quality (same output+distribution). SpeculativeDecoder: draft_size+target_size+num_speculative+accept_rate+draft_tokens(num)+verify (random < rate)+step (draft+verify)+speedup_factor (sequential/parallel). Ventajas speculative vs sequential: speedup (2-3x+parallel), no quality (same+distribution), latency (reduce+TTFT), same output (deterministic+verified), cost (efficient+less compute). Criterios: EAGLE3 = draft model+SOTA+train, Medusa = heads+fast+self, N-gram = simple+no train+cheap, Lookahead = Jacobi+parallel. Decision: SOTA -> EAGLE3, fast -> Medusa, simple -> n-gram, parallel -> Lookahead. Frameworks: vllm, sglang, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + decoding.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SpeculativeDecoder con draft + target.
- Implementar draft_tokens + verify.
- Implementar step + speedup_factor.
- Diagnosticar advantages.
- Diagnosticar EAGLE3 vs Medusa vs n-gram.

## Constrúyelo

```python
class SpeculativeDecoder:
    def speedup_factor(self):
        sequential_tokens = self.num_speculative
        parallel_tokens = 1 + self.accept_rate * self.num_speculative
        if parallel_tokens == 0:
            return 1.0
        return sequential_tokens / parallel_tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: eagle3-speculative
fase: 17
leccion: 05
---

1. SpeculativeDecoder.
2. draft + verify.
3. speedup.
4. +Production.
```

## Ejercicios

1. **Speculative**: probar
   accept_rate.
2. **Speedup**: probar
   factor.
3. **Desafio**: integrar
   con vLLM.

## Lecturas recomendadas

- "EAGLE" (Li, 2024)
- "Speculative Decoding" (Leviathan, 2023)
- "vLLM Speculative" (vLLM, 2024)

---

> 📚 **Adaptación al español de la lección [EAGLE3 Speculative Decoding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).