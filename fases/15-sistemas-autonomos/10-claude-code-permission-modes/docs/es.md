# Claude Code permission modes

> Claude Code permission modes: (1) default: bash+write+edit requieren approval; read+search+grep+glob no, (2) acceptEdits: write+edit auto-aceptados; bash+approve, (3) plan: plan mode bloquea tools (* requiere), (4) dontAsk: deny by default, (5) bypassPermissions: skip checks. Cuándo usar plan mode: tasks largas, multi-file, dangerous (DB/FS), review. dontAsk vs bypassPermissions: dontAsk = deny by default + no ejecuta + safe; bypassPermissions = auto-acepta + ejecuta + riesgo. Patrón: define modes dict con name+description+tools_require_approval+tools_no_approval+deny_by_default, requires_approval function, plan mode con steps+user approve, dontAsk default deny, bypassPermissions skip. Decision: default -> production, acceptEdits -> dev, plan -> tasks largas, dontAsk -> read-only, bypassPermissions -> trusted. Frameworks: anthropic. +Production, +Safe, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PERMISSION_MODES dict con 5 modes.
- Implementar list_permission_modes + get_mode.
- Implementar requires_approval con * + deny_by_default.
- Implementar plan_step.
- Diagnosticar modes.

## Constrúyelo

```python
PERMISSION_MODES = {
    "default": {
        "name": "default",
        "tools_require_approval": ["bash", "write", "edit"],
        "tools_no_approval": ["read", "search", "grep", "glob"],
    },
    # ... 5 modes
}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: permission-modes
fase: 15
leccion: 10
---

1. default + acceptEdits.
2. plan + dontAsk.
3. bypassPermissions.
4. +Production.
```

## Ejercicios

1. **Permission modes**: probar
   los 5 modes.
2. **Plan mode**: implementar
   un plan mode real.
3. **Desafio**: integrar
   permission modes con tu agent.

## Lecturas recomendadas

- "Claude Code: Permission Modes" (Anthropic, 2024)
- "Anthropic SDK: Permission API" (Anthropic, 2024)
- "Anthropic SDK" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Claude Code Permission Modes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).