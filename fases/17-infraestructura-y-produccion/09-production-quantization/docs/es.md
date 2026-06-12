# Production quantization

> Quant formats: (1) INT8 (PTQ+2x), (2) INT4 (GPTQ/AWQ+4x), (3) FP8 (Native+H100), (4) FP4 (NVFP4+B200), (5) GPTQ+AWQ+GGUF. QUANT_FORMATS: FP16/BF16 (16 bits), INT8 (8 bits 2x), FP8 (8 bits 2x), INT4 (4 bits 4x), FP4 (4 bits 4x), GGUF (Q4/Q5/Q8)+bits+method+compression+accuracy_loss. estimate_size: bits/8 * model_size. pick_format: target_compression+max_loss+min accuracy_loss. Diferencias: PTQ = post-train+fast+cheap, QAT = train-aware+accurate+slow, GPTQ = groups+accurate+low memory, AWQ = activation-aware+accurate+mixed. Criterios: INT4 = max compression+4x+loss, INT8 = balance+2x+reliable, FP8 = H100++2x+native, FP4 = B200+4x+new. Decision: max -> INT4, balance -> INT8, H100 -> FP8, B200 -> FP4. Frameworks: vllm, nvidia, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + quant.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar QUANT_FORMATS con 10 formats.
- Implementar estimate_size con bits/8.
- Implementar pick_format con target + max_loss.
- Diagnosticar PTQ vs QAT vs GPTQ vs AWQ.
- Diagnosticar INT4 vs INT8 vs FP8 vs FP4.

## Constrúyelo

```python
QUANT_FORMATS = {
    "INT4": {"bits": 4, "compression": 4.0, "accuracy_loss": 0.02},
    "INT8": {"bits": 8, "compression": 2.0, "accuracy_loss": 0.005},
    # ... 10 formats
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
name: production-quantization
fase: 17
leccion: 09
---

1. 10 formats.
2. estimate_size.
3. pick_format.
4. +Production.
```

## Ejercicios

1. **Formats**: probar
   los 10.
2. **Size**: probar
   estimate.
3. **Desafio**: integrar
   con vLLM + GPTQ.

## Lecturas recomendadas

- "GPTQ" (Frantar, 2023)
- "AWQ" (Lin, 2023)
- "FP8 Training" (NVIDIA, 2024)

---

> 📚 **Adaptación al español de la lección [Production Quantization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).