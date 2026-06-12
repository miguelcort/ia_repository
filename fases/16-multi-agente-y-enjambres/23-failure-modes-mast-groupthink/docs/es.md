# Failure modes MAST groupthink

> MAST: (1) Verification (skip checks+false positive), (2) Task decomp (wrong subtasks+missing), (3) Multi-agent (groupthink+cascade+free-riding+signaling), (4) System (conflicting+unclear role). Groupthink+cascade+free-riding+signaling. groupthink: ratio (most_common/total)+diversity (unique/total)+threshold (0.8+0.3). free_riding: avg mean+< avg * threshold. cascade: matches count+> threshold. deadlock: wait_steps per agent+> max. Ventajas MAST vs ad-hoc: standard (common+reference), comprehensive (14 cats+coverage), comparable (across systems+benchmark), reproducible (same+repeat), mitigations (per cat+known). Criterios: MAST = standard+comprehensive+14 cats, Custom = specific+tailored+domain, Ad-hoc = quick+cheap+limited. Decision: standard -> MAST, specific -> custom, quick -> ad-hoc, mix -> MAST+custom. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/22
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MAST_CATEGORIES con 4 cats.
- Implementar detect_groupthink + detect_free_riding.
- Implementar detect_cascade + detect_deadlock.
- Diagnosticar MAST.
- Diagnosticar MAST vs ad-hoc.

## Constrúyelo

```python
def detect_groupthink(agents, decisions, threshold=0.8):
    if not decisions:
        return False
    counts = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    most_common = max(counts, key=counts.get)
    ratio = counts[most_common] / len(decisions)
    diversity = len(set(decisions)) / max(len(decisions), 1)
    if ratio > threshold and diversity < 0.3:
        return True
    return False
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: failure-modes-mast
fase: 16
leccion: 23
---

1. MAST + 4 cats.
2. groupthink + free-riding.
3. cascade + deadlock.
4. +Production.
```

## Ejercicios

1. **Groupthink**: probar
   con 4 decisions.
2. **Free-riding**: probar
   threshold.
3. **Desafio**: integrar
   con monitoring.

## Lecturas recomendadas

- "MAST" (Merrill, 2024)
- "Groupthink" (Janis, 1972)
- "Multi-Agent Failures" (Han, 2024)

---

> 📚 **Adaptación al español de la lección [Failure Modes MAST Groupthink]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).