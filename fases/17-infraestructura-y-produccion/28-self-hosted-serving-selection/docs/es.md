# Self-hosted serving selection

> Serving: (1) vLLM (PagedAttn+high), (2) TGI (HF+easy), (3) TensorRT-LLM (NV+very high), (4) LMDeploy (multi+high), (5) llama.cpp (CPU+edge). SERVING_OPTIONS: vLLM (Python+GPU+high+medium+2023)+TGI (Rust/Python+GPU+high+high+2023)+TensorRT-LLM (C++/Python+GPU+very_high+low+2024)+LMDeploy (Python/C+++GPU+high+medium+2024)+llama.cpp (C+++no GPU+medium+medium+2023). by_throughput: order (low/medium/high/very_high)+threshold. recommend: have_gpu+ease+min_throughput. Criterios: vLLM = PagedAttn+high throughput+OSS, TGI = HF+simple+multi-model, TRT-LLM = NV+max perf+compile, llama.cpp = CPU+edge+no GPU. Decision: PagedAttn -> vLLM, HF -> TGI, NV -> TRT-LLM, CPU -> llama. Criterios self-hosted vs managed: Self-hosted = control+cost+privacy, Managed = scale+no ops+latest, Hybrid = tiered+best of both+fallback. Decision: control -> self, scale -> managed, mix -> hybrid. Frameworks: vllm, tgi, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + serving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/27
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SERVING_OPTIONS con 5 options.
- Implementar by_throughput.
- Implementar recommend con constraints.
- Diagnosticar selection.
- Diagnosticar self-hosted vs managed.

## Constrúyelo

```python
SERVING_OPTIONS = {
    "vllm": {"name": "vLLM", "language": "Python", "gpu_required": True, "throughput": "high", "ease_of_use": "medium"},
    "tgi": {"name": "TGI", "language": "Rust/Python", "gpu_required": True, "throughput": "high", "ease_of_use": "high"},
    "tensorrt_llm": {"name": "TensorRT-LLM", "language": "C++/Python", "gpu_required": True, "throughput": "very_high", "ease_of_use": "low"},
    "llama_cpp": {"name": "llama.cpp", "language": "C++", "gpu_required": False, "throughput": "medium", "ease_of_use": "medium"},
    # ...
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
name: self-hosted-serving
fase: 17
leccion: 28
---

1. 5 serving options.
2. by_throughput.
3. recommend.
4. +Production.
```

## Ejercicios

1. **Options**: probar
   los 5.
2. **Recommend**: probar
   constraints.
3. **Desafio**: integrar
   con vLLM + K8s.

## Lecturas recomendadas

- "vLLM" (Kwon, 2023)
- "TGI" (HuggingFace, 2024)
- "TensorRT-LLM" (NVIDIA, 2024)

---

> 📚 **Adaptación al español de la lección [Self-hosted Serving Selection]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).