# Disaggregated prefill decode

> Disagg prefill-decode: (1) Separate (GPUs), (2) Prefill (compute+batched), (3) Decode (memory+sequential), (4) Mooncake+DistServe (architectures), (5) Throughput (high+latency low). PrefillEngine: name+compute_tflops+queue+busy+prefill (check busy+return kv). DecodeEngine: name+memory_gb+active+decode_step (return next_token+kv_state). DisaggregatedEngine: prefill+decode+generate (prefill+loop decode). Diferencias: Prefill = compute-bound+parallel+batch+fast+TTFT, Decode = memory-bound+sequential+per-req+KV+TPOT. Criterios: Disagg = max throughput+scale+complex, Colocation = simple+cheap+dev, Splitwise = prefix+decode+mixed. Decision: max -> disagg, simple -> colocate, mixed -> splitwise, mix -> all. Frameworks: vllm, mooncake, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + disagg.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PrefillEngine con compute_tflops + busy.
- Implementar DecodeEngine con memory_gb.
- Implementar DisaggregatedEngine con prefill + decode.
- Diagnosticar prefill vs decode.
- Diagnosticar disagg vs colocate.

## Constrúyelo

```python
class DisaggregatedEngine:
    def generate(self, prompt_tokens, max_tokens=10):
        kv = self.prefill.prefill(prompt_tokens)
        if kv is None:
            return None
        results = []
        for _ in range(max_tokens):
            step = self.decode.decode_step(kv)
            results.append(step["next_token"])
            kv = step["kv_state"]
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
name: disagg-prefill-decode
fase: 17
leccion: 17
---

1. Prefill + Decode.
2. Disaggregated.
3. generate.
4. +Production.
```

## Ejercicios

1. **Prefill**: probar
   busy.
2. **Decode**: probar
   step.
3. **Desafio**: integrar
   con vLLM disagg.

## Lecturas recomendadas

- "DistServe" (Zhong, 2024)
- "Mooncake" (Moonshot, 2024)
- "Splitwise" (Patel, 2024)

---

> 📚 **Adaptación al español de la lección [Disaggregated Prefill Decode]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).