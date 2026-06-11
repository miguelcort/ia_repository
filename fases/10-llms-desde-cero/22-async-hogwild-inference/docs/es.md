# Async Hogwild inference

> Async inference patterns: async/await + batching (5-10x throughput), continuous batching (vLLM, SGLang, TGI, 2-4x vs static), Hogwild (no coordination, multiple workers, simple, +memory cost). Métricas SOTA: tokens/sec, requests/sec, TTFT (Time To First Token, prefill latency), TPOT (Time Per Output Token, decode latency), total latency. Benchmarks: vLLM Serve, llm-perf, GenAI-PerF, mlperf inference. SOTA: Llama 70B 50+ tokens/sec A100, TTFT <100ms, TPOT <20ms. Hoy: vLLM/SGLang + continuous + spec decode + INT4 + paged = gold standard. Frontier: 1000+ tok/sec 70B en 1 GPU con custom hardware (Groq, Cerebras, SambaNova).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/12-optimizacion-de-inferencia
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar async batch simulation.
- Calcular throughput sync vs async.
- Diagnosticar Hogwild.
- Medir throughput metrics.

## Constrúyelo

```python
def throughput_async(n_requests, batch_size=4, latency_per_batch=120):
    n_batches = n_requests / batch_size
    return n_requests / (n_batches * latency_per_batch / 1000)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-async-inference
fase: 10
leccion: 22
---

1. Async + batching.
2. Continuous batching.
3. Hogwild no coord.
4. Tokens/sec, TTFT, TPOT.
5. vLLM, SGLang, TGI.
```

## Ejercicios

1. **Async**: implementar async
   batch con asyncio.
2. **vLLM**: deploy Llama 3
   con vLLM continuous batching.
3. **Desafio**: medir TTFT
   y TPOT de tu model.

## Lecturas recomendadas

- "vLLM: Efficient Memory Management for LLM Serving" (Kwon et al., 2023)
- "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al., 2024)
- "LLM Inference Unveiled" (Yuan et al., 2024)
- "How to Scale Your LLM Inference" (vLLM, 2024)

---

> 📚 **Adaptación al español** de la lección "[Async Hogwild Inference]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).