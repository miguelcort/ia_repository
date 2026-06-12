# Why models fail

> Por que fallan los modelos en agent workbench: (1) Ambiguity (multi-interpret+unclear), (2) Missing context (no files+no docs), (3) Scope creep (more/less+drift), (4) Tool errors (external+retry), (5) Infinite loop (repeat+no progress), (6) Instruction drift (forget+long ctx). Deteccion: Ambiguity (?, or, maybe, perhaps), Missing (404, not found, no such file), Tool errors (error, exception, traceback), Loop (step 50, same as before, loop), Drift (ignore previous, earlier you said). Mitigations: Ambiguity -> ask clarification+examples, Missing -> context+attach files+init, Scope -> contracts+reviewer, Tools -> validate+retry+fallback, Loop -> max steps+cycle detect, Drift -> recap+structured. Criterios: Mitigations = critical+repeated+production, Self-correct = simple+rare+dev. Decision: critical -> mitigations, simple -> self, mix -> both, production -> both. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + reliability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/30
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar FAILURE_MODES dict con 6 modes.
- Implementar list_failure_modes + get_failure_mode.
- Implementar detect_failure_mode + detect_all_failure_modes.
- Implementar suggest_mitigations.
- Diagnosticar mitigations.

## Constrúyelo

```python
FAILURE_MODES = {
    "ambiguity": {
        "name": "Ambiguity",
        "signals": ["?", "or", "maybe"],
        "mitigation": "Ask for clarification, request examples",
    },
    # ... 6 modes
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
name: why-models-fail
fase: 14
leccion: 31
---

1. 6 failure modes.
2. detect + signals.
3. mitigations.
4. +Production.
```

## Ejercicios

1. **Failure modes**: probar
   los 6 modes.
2. **Detect**: probar
   signals matching.
3. **Desafio**: integrar
   con un agent real.

## Lecturas recomendadas

- "Why Language Models Hallucinate" (OpenAI, 2024)
- "Anthropic: Claude Reliability" (Anthropic, 2024)
- "Agent Failure Modes" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Why Models Fail]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).