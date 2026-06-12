# Checkpoints & rollback

> Checkpoints + rollback: (1) Snapshot state (copy+ID), (2) Restore on failure (load+recovery), (3) Multi-step recovery (latest+named), (4) Durable storage (disk+DB), (5) Versioned checkpoints (metadata+history). CheckpointStore: checkpoints dict (ID+state+metadata+time), history list (order+latest), save(state, metadata) UUID+deep copy, load(cp_id) deep copy+KeyError si no, list() ID+time, delete(cp_id) pop, latest() last+None. Rollbackable: state dict + checkpoint(name) save+rollback(cp_id) load+rollback_latest() latest+ValueError. with_checkpoint: try fn() + except + load latest + state.clear() + state.update(load) + raise re-raise+propagate. diff_states: keys union + iterate sorted + append (k, before, after) si difieren. Criterios: checkpoints = explicit+state+snapshot, durable = implicit+workflow+replay. Decision: state -> checkpoints, workflow -> durable, mix -> ambos, production -> mix. Frameworks: temporal, dbos, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + durability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/15
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar CheckpointStore con save + load + list + delete + latest.
- Implementar Rollbackable con checkpoint + rollback + rollback_latest.
- Implementar with_checkpoint helper.
- Implementar diff_states.
- Diagnosticar checkpoints vs durable.

## Constrúyelo

```python
class CheckpointStore:
    def save(self, state, metadata=None):
        cp_id = str(uuid.uuid4())
        self.checkpoints[cp_id] = {"id": cp_id, "state": copy.deepcopy(state), "metadata": metadata or {}}
        self.history.append(cp_id)
        return cp_id

    def load(self, cp_id):
        return copy.deepcopy(self.checkpoints[cp_id]["state"])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: checkpoints-rollback
fase: 15
leccion: 16
---

1. CheckpointStore + save / load.
2. Rollbackable + rollback.
3. with_checkpoint + diff_states.
4. +Production.
```

## Ejercicios

1. **CheckpointStore**: probar
   save + load + deepcopy.
2. **with_checkpoint**: probar
   rollback on failure.
3. **Desafio**: integrar
   con Temporal o DBOS.

## Lecturas recomendadas

- "Temporal: Checkpointing" (Temporal, 2019)
- "DBOS: Checkpoints" (DBOS, 2024)
- "Checkpoint/Restore" (Linux CRIU, 2024)

---

> 📚 **Adaptación al español de la lección [Checkpoints & Rollback]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).