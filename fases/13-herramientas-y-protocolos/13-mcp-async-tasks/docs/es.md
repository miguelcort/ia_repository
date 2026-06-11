# MCP async tasks

> MCP async tasks: long-running operations. Methods: task/create (start, return task_id), task/get (status, result, error), task/list (enumerate, filter by status), task/cancel (abort, -if completed). Polling vs streaming: (a) polling periodic task/get, +simple +compatible -latency, (b) streaming SSE updates +real-time +efficient +latency. Status state machine: pending (created, not started) -> running (in progress) -> completed (success, result available) | failed (error, error message) | cancelled (aborted, user cancelled). +Long-running, +Async, +Cancelable, +Resilient, +Reliable. Frameworks: mcp, fastmcp, anthropic, openai, langchain, asyncio, aiohttp. +Production: standard 2024-25. +Use cases: agent, RAG, automation, long ops, real-time. Decision: simple -> polling, real-time -> streaming, production -> streaming. Trade-offs: async + flexible, sync + simple, polling + simple, streaming + real-time. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Task class.
- Implementar TaskManager con create/get/list/cancel.
- Implementar mark_running/completed/failed.
- Diagnosticar polling vs streaming.
- Diagnosticar status state machine.

## Constrúyelo

```python
class TaskManager:
    def create_task(self, operation, params=None):
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        task = Task(task_id, operation, params)
        self.tasks[task_id] = task
        return task
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-async-tasks
fase: 13
leccion: 13
---

1. task/create.
2. task/get.
3. task/list.
4. task/cancel.
5. State machine.
```

## Ejercicios

1. **Async tasks**: implementar
   long-running con asyncio.
2. **Polling**: probar polling
   vs streaming.
3. **Desafio**: custom
   async tasks.

## Lecturas recomendadas

- "MCP Async Tasks Specification" (Anthropic, 2024)
- "Async Python" (asyncio Documentation)
- "Server-Sent Events for Tasks" (Anthropic, 2024)
- "Task State Machines" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Async Tasks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).