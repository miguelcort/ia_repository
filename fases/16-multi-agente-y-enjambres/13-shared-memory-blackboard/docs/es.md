# Shared memory blackboard

> Blackboard: (1) Shared state (multi-agent+common), (2) Key-value (Dict+simple), (3) Locks (mutex+critical section), (4) Version (counter+optimistic), (5) Access log (audit+history), (6) Pattern queries (regex+filter). Blackboard: _data Dict+_version Dict+_log List+_locks Set+write(key, value, author) check lock+increment version+read(key) get+version(key) counter+lock(key, author) add to set+unlock(key, author) remove+pattern_query(pattern) regex+access_log(key) filter. Ventajas version+lock vs plain dict: optimistic concurrency (version check+no overwrite), audit (who wrote+when), conflict detection (detect+resolve), rollback (restore+version history), pattern (regex+bulk). Criterios: Blackboard = shared+query+sync, Pub/Sub = events+1-N+subscribe, Queue = tasks+async+workers. Decision: shared -> blackboard, events -> pub/sub, tasks -> queue, mix -> blackboard+pub/sub. Frameworks: langchain, crewai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + blackboard.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/12
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Blackboard con _data + _version + _log + _locks.
- Implementar write + read + version.
- Implementar lock + unlock.
- Implementar pattern_query.
- Implementar access_log.
- Diagnosticar blackboard vs pub/sub.

## Constrúyelo

```python
class Blackboard:
    def write(self, key, value, author):
        if key in self._locks:
            raise RuntimeError(f"key {key} is locked")
        self._data[key] = value
        self._version[key] = self._version.get(key, 0) + 1
        self._log.append({"op": "write", "key": key, "author": author, "ts": time.time()})
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: shared-memory-blackboard
fase: 16
leccion: 13
---

1. Blackboard.
2. write + lock + version.
3. pattern_query + log.
4. +Production.
```

## Ejercicios

1. **Blackboard**: probar
   write + lock.
2. **pattern_query**: probar
   regex.
3. **Desafio**: integrar
   con CrewAI memory.

## Lecturas recomendadas

- "Blackboard Systems" (Engelmore, 1988)
- "Distributed Shared Memory" (Tanenbaum, 2007)
- "CrewAI: Memory" (CrewAI, 2024)

---

> 📚 **Adaptación al español de la lección [Shared Memory Blackboard]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).