# Cuantización

> PTQ (post-training): GPTQ, AWQ, SmoothQuant, bitsandbytes. QAT: quantization-aware training. INT8 (2x), INT4 (4x), FP8 (H100 native, E4M3/E5M2, 1.5x), NF4 (QLoRA). Symmetric (absmax), asymmetric (minmax + zero point). Per-tensor, per-channel, per-group. GPTQ: error compensation columna por columna con Hessian. AWQ: activation-aware weight quantization, protege top 1% salient weights, scale per-channel. Frameworks: AutoAWQ, AutoGPTQ, bitsandbytes, torchao, llama.cpp, SGLang. Hoy: INT4 (AWQ/GPTQ) inference, FP8 training, NF4 fine-tune.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar absmax quantization.
- Implementar NF4 quantization.
- Implementar GPTQ-style per-row.
- Calcular memory savings.
- Diagnosticar métodos SOTA.

## Constrúyelo

```python
def absmax_quantize(x, num_bits=8):
    abs_max = np.abs(x).max()
    scale = (2 ** (num_bits - 1) - 1) / abs_max
    x_q = np.round(x * scale).clip(-128, 127)
    return x_q.astype(np.int8), 1.0 / scale
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-quantization
fase: 10
leccion: 11
---

1. PTQ, QAT.
2. INT8, INT4, FP8, NF4.
3. GPTQ error compensation.
4. AWQ activation-aware.
5. AutoAWQ, bitsandbytes.
```

## Ejercicios

1. **GPTQ**: implementar GPTQ
   simple para weight matrix.
2. **AWQ**: aplicar a Llama 2 7B
   con AutoAWQ.
3. **Desafio**: comparar FP8
   vs BF16 training.

## Lecturas recomendadas

- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)
- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2023)
- "FP8 Formats for Deep Learning" (NVIDIA, 2022)

---

> 📚 **Adaptación al español** de la lección "[Quantization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).