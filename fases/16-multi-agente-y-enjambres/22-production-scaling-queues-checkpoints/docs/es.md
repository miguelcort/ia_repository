# Production scaling queues checkpoints

> Production: (1) Queues (backpressure+max size), (2) Horizontal (workers+scale up/down), (3) Checkpoints (save+restore), (4) Graceful (drain+stop), (5) Monitoring (metrics+alert). Queue: max_size+items+put (check full+drop if full)+get (pop 0)+full (>=max)+dropped counter. Worker: agent_id+queue+step (get+process)+start+stop. horizontal_scale: thresholds+add+remove. graceful_shutdown: stop workers+drain. CheckpointStore: save(agent_id, state)+load(agent_id). Ventajas horizontal vs vertical: scale (add nodes+linear), fault tolerance (one fails+redundancy), cost (commodity+cheap), elasticity (auto+dynamic), geographic (multi-region+latency). Criterios: Queue = tasks+async+workers, Pub/Sub = events+1-N+subscribe, In-memory = fast+ephemeral+per-process. Decision: tasks -> queue, events -> pub/sub, fast -> in-mem, mix -> queue+pub/sub. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + production.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/21
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar Queue con put + get + full + dropped.
- Implementar Worker con step + start + stop.
- Implementar horizontal_scale con up/down.
- Implementar graceful_shutdown con drain.
- Implementar CheckpointStore con save/load.
- Diagnosticar horizontal vs vertical.

## Constrúyelo

```python
class Queue:
    def put(self, item, timeout=None):
        if len(self.items) >= self.max_size:
            self.dropped += 1
            return False
        self.items.append(item)
        return True
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: production-scaling
fase: 16
leccion: 22
---

1. Queue + Worker.
2. horizontal_scale.
3. graceful_shutdown.
4. CheckpointStore.
5. +Production.
```

## Ejercicios

1. **Queue**: probar
   put + drop.
2. **Worker**: probar
   step + stop.
3. **Desafio**: integrar
   con Temporal.

## Lecturas recomendadas

- "Distributed Systems" (Tanenbaum, 2007)
- "Kafka" (Kreps, 2011)
- "Production-Ready Microservices" (Richardson, 2016)

---

> 📚 **Adaptación al español de la lección [Production Scaling Queues Checkpoints]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).