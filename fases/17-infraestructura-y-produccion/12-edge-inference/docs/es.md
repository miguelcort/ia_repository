# Edge inference

> Edge inference: (1) On-device (local+privacy), (2) Mobile+IoT (embedded), (3) Compression (INT4+small), (4) Latency (low+real-time), (5) Cloud fallback (hybrid). EdgeDevice: memory_mb+compute_score+supports_quantization+running_model+can_run (model_size, requires_quant)+load (name, size)+unload. EdgeRouter: devices list+cloud_endpoint+route (model, size, quant) iterate devices+cloud fallback. estimate_latency: size/compute*tokens*0.01. Ventajas edge vs cloud: latency (local+real-time), privacy (on-device+no upload), offline (no network+always), cost (no API+amortize), bandwidth (local+no transfer). Criterios: Edge = latency+privacy+offline, Cloud = scale+latest+big model, Hybrid = tiered+best of both+fallback. Decision: latency -> edge, scale -> cloud, mix -> hybrid, mix -> edge+cloud. Frameworks: llama.cpp, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + edge.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/11
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar EdgeDevice con memory + compute + quant.
- Implementar can_run + load + unload.
- Implementar EdgeRouter con devices + cloud.
- Implementar route + cloud fallback.
- Implementar estimate_latency.
- Diagnosticar edge vs cloud.

## Constrúyelo

```python
class EdgeDevice:
    def can_run(self, model_size_mb, requires_quantization=False):
        if requires_quantization and not self.supports_quantization:
            return False
        return self.memory_mb >= model_size_mb
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: edge-inference
fase: 17
leccion: 12
---

1. EdgeDevice.
2. EdgeRouter.
3. can_run + load.
4. estimate_latency.
5. +Production.
```

## Ejercicios

1. **EdgeDevice**: probar
   can_run.
2. **Router**: probar
   route + fallback.
3. **Desafio**: integrar
   con llama.cpp.

## Lecturas recomendadas

- "llama.cpp" (ggerganov, 2024)
- "MLC LLM" (MLC, 2024)
- "Edge AI" (Microsoft, 2024)

---

> 📚 **Adaptación al español de la lección [Edge Inference]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).