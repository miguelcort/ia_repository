# Runtime feedback loops

> Feedback loops: (1) Observe (state+keys), (2) Detect (keywords+errors), (3) Recover (strategy+fn), (4) Retry (backoff+max), (5) Escalate (level+message), (6) Long-running (agents+resilient). FeedbackLoop: max_retries+backoff+on_error, run(fn) try/except+backoff ** attempt, history (list+attempt+ok), last_attempts(n) [-n:]. observe_state: check expected_keys+missing list, return (ok, missing). detect_error: keyword match (error/exception/failed/fatal/panic), return error type+None si no. Recovery: strategies dict+register+recover (lookup+call). escalate: level+message+timestamp. Criterios: Feedback loops = retry+recover+self-heal, Exceptions = surface+bubble up+crash, Supervisor = monitor+restart+external. Decision: retry -> feedback, surface -> exceptions, monitor -> supervisor, mix -> all. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + resilience.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/36
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar FeedbackLoop con max_retries + backoff.
- Implementar observe_state + detect_error.
- Implementar Recovery + escalate.
- Diagnosticar retry vs exception vs supervisor.
- Diagnosticar feedback loops vs other.

## Constrúyelo

```python
class FeedbackLoop:
    def run(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                self.history.append({"attempt": attempt, "ok": True, "result": result})
                return result
            except Exception as e:
                last_exc = e
                self.history.append({"attempt": attempt, "ok": False, "error": str(e)})
                if self.on_error:
                    self.on_error(e, attempt)
                if attempt < self.max_retries:
                    time.sleep(self.backoff ** attempt * 0.01)
        raise last_exc
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: runtime-feedback-loops
fase: 14
leccion: 37
---

1. FeedbackLoop + retry.
2. observe + detect.
3. Recovery + escalate.
4. +Production.
```

## Ejercicios

1. **FeedbackLoop**: probar
   retry + backoff.
2. **observe_state**: probar
   missing keys.
3. **Desafio**: integrar
   con tu agent.

## Lecturas recomendadas

- "Anthropic: Claude Retry" (Anthropic, 2024)
- "OpenAI: Error Handling" (OpenAI, 2024)
- "Feedback Loops" (Hogarth, 2024)

---

> 📚 **Adaptación al español de la lección [Runtime Feedback Loops]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).