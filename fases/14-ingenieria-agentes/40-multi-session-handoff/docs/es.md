# Multi-session handoff

> Multi-session handoff: (1) State transfer (context+transcript), (2) Resume points (last_active+checkpoint), (3) Session ids (UUID+unique), (4) Context carryover (inherit+snapshot), (5) Subset (selective+filter). Session: session_id (UUID) + context (Dict) + transcript (List) + update(key, val) last_active + append(role, content) ts + snapshot() Dict + from_snapshot(snap) Restore. HandoffRegistry: sessions dict+create()+get(id)+handoff(src, subset) Inherit+Filter. Ventajas handoff subset vs full copy: selective context (filter+relevant), privacy (PII redaction+scope), size (smaller+faster), speed (transfer+load), audit (minimal+diff). Criterios: Handoff = long+resume+multi-user+audit, Single = short+ephemeral+simple+fast. Decision: long -> handoff, short -> single, mix -> both, production -> mix. Frameworks: langchain, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + state.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/39
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Session con context + transcript + snapshot.
- Implementar HandoffRegistry con handoff + subset.
- Diagnosticar full vs subset.
- Diagnosticar handoff vs single.

## Constrúyelo

```python
def handoff(self, session_id, context_subset=None):
    old = self.sessions.get(session_id)
    new_context = dict(old.context)
    if context_subset:
        new_context = {k: old.context.get(k) for k in context_subset if k in old.context}
    new = self.create(context=new_context)
    new.append("handoff", f"from {session_id}")
    return new
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: multi-session-handoff
fase: 14
leccion: 40
---

1. Session + snapshot.
2. HandoffRegistry.
3. handoff subset.
4. +Production.
```

## Ejercicios

1. **Session**: probar
   snapshot + restore.
2. **Handoff**: probar
   full + subset.
3. **Desafio**: integrar
   con checkpointing.

## Lecturas recomendadas

- "Anthropic: Multi-Turn" (Anthropic, 2024)
- "OpenAI: Threads" (OpenAI, 2024)
- "LangChain: Memory" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Multi-Session Handoff]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).