# TensorRT-LLM Blackwell

> TensorRT-LLM: (1) NVIDIA opt (compile+tune), (2) Kernel fusion (reduce ops+speed), (3) FP4/FP8 (quantize+memory), (4) In-flight (continuous+throughput), (5) Blackwell (B100/B200+Hopper). NVIDIA_GPUS: A100 (Ampere), H100 (Hopper), H200 (Hopper 141GB), B100 (Blackwell 192GB+3600 fp8+864 cores), B200 (Blackwell 192GB+4500 fp8+1080 cores). best_for: max fp8_tflops+max memory. can_run_model: bytes per param (2 for new, 4 for old)+memory check. Blackwell specs: B100 192GB/3600 fp8/864 cores, B200 192GB/4500 fp8/1080 cores. Criterios: TensorRT-LLM = NVIDIA+SOTA+compile, vLLM = PagedAttn+multi-GPU+OSS, Raw = simple+dev+cheap. Decision: NVIDIA -> TRT, multi-GPU -> vLLM, dev -> raw, mix -> TRT+vLLM. Frameworks: nvidia, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + nvidia.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar NVIDIA_GPUS con 5 GPUs.
- Implementar best_for_throughput + best_for_memory.
- Implementar can_run_model.
- Diagnosticar Blackwell specs.
- Diagnosticar TRT vs vLLM vs raw.

## Constrúyelo

```python
def can_run_model(gpu_name, model_size_b):
    gpu = NVIDIA_GPUS.get(gpu_name)
    if not gpu:
        return False
    bytes_per_param = 2 if "H" in gpu["architecture"] or "B" in gpu["architecture"] else 4
    required_gb = model_size_b * bytes_per_param
    return gpu["memory_gb"] >= required_gb
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: tensorrt-llm-blackwell
fase: 17
leccion: 07
---

1. NVIDIA_GPUS.
2. best_for.
3. can_run_model.
4. Blackwell.
5. +Production.
```

## Ejercicios

1. **GPUs**: probar
   los 5.
2. **Best**: probar
   throughput.
3. **Desafio**: integrar
   con TensorRT-LLM.

## Lecturas recomendadas

- "TensorRT-LLM" (NVIDIA, 2024)
- "Blackwell GPU" (NVIDIA, 2024)
- "FP8 Training" (NVIDIA, 2024)

---

> 📚 **Adaptación al español de la lección [TensorRT-LLM Blackwell]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).