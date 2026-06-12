# Batch APIs

> Batch APIs: (1) OpenAI/Anthropic (built-in), (2) 50% (discount), (3) Async 24h (SLA), (4) JSONL (input), (5) Bulk (eval+dataset). BatchRequest: custom_id+method+url+body. BatchJob: requests list+add+to_jsonl (JSON lines)+total+estimated_cost (per_request, discount)+mark_complete. parse_jsonl_results(text): lines -> dicts. Ventajas batch vs sync: 50% cost (discount+cheap), bulk (many+pipeline), JSONL (simple+file), async (no wait+continue), 24h SLA (predictable). Criterios: Batch = bulk+OK wait+cost, Sync = instant+single+real-time, Streaming = tokens+real-time+UI. Decision: bulk -> batch, single -> sync, real-time -> streaming, mix -> sync+batch. Frameworks: openai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + batch.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar BatchRequest con custom_id + method + url + body.
- Implementar BatchJob con requests.
- Implementar add + to_jsonl.
- Implementar total + estimated_cost.
- Diagnosticar batch vs sync.

## Constrúyelo

```python
class BatchJob:
    def to_jsonl(self):
        lines = []
        for r in self.requests:
            lines.append(json.dumps({
                "custom_id": r.custom_id,
                "method": r.method,
                "url": r.url,
                "body": r.body,
            }))
        return "\n".join(lines)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: batch-apis
fase: 17
leccion: 15
---

1. BatchRequest.
2. BatchJob + JSONL.
3. estimated_cost.
4. +Production.
```

## Ejercicios

1. **Job**: probar
   to_jsonl.
2. **Cost**: probar
   estimated.
3. **Desafio**: integrar
   con OpenAI Batch API.

## Lecturas recomendadas

- "OpenAI Batch API" (OpenAI, 2024)
- "Anthropic Messages Batches" (Anthropic, 2024)
- "JSONL Format" (JSON Lines, 2024)

---

> 📚 **Adaptación al español de la lección [Batch APIs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).