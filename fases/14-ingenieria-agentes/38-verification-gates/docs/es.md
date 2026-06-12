# Verification gates

> Verification gates: (1) Pre/post (pre-deploy+post-deploy), (2) Unit tests (pytest+coverage), (3) Lint (ruff+flake8), (4) Type check (mypy+pyright), (5) Schema validation (JSON Schema+Pydantic), (6) Gate sequence (compose+fail_fast). Gate: name+fn+description+run() try/except+last_result. GateSequence: gates list+fail_fast+add(g)+run() iterate+history+all_passed() all(). Helpers: run_unit_tests (path+passed+tests), lint_check (path+issues), type_check (path+errors), contract_check (typ+val+isinstance), schema_validate (schema+data+missing+type). Criterios: Gates = pre-deploy+post-deploy+runtime, CI = per-PR+per-merge+async, Pre-commit = per-commit+local+fast. Decision: pre-deploy -> gates, per-PR -> CI, per-commit -> pre-commit, mix -> all. Frameworks: pytest, mypy, ruff, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + quality.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/37
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Gate con name + fn + description + run.
- Implementar GateSequence con add + run + fail_fast + all_passed.
- Implementar helpers: unit + lint + type + contract + schema.
- Diagnosticar fail_fast.
- Diagnosticar gates vs CI vs pre-commit.

## Constrúyelo

```python
class Gate:
    def run(self, *args, **kwargs):
        try:
            result = self.fn(*args, **kwargs)
            self.last_result = {"ok": True, "result": result}
        except Exception as e:
            self.last_result = {"ok": False, "error": str(e)}
        return self.last_result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: verification-gates
fase: 14
leccion: 38
---

1. Gate + GateSequence.
2. unit + lint + type.
3. contract + schema.
4. +Production.
```

## Ejercicios

1. **Gate**: probar
   run + last_result.
2. **GateSequence**: probar
   fail_fast.
3. **Desafio**: integrar
   con pre-commit.

## Lecturas recomendadas

- "Pre-commit Hooks" (pre-commit, 2024)
- "GitHub Actions" (GitHub, 2024)
- "Pydantic: Validation" (Pydantic, 2024)

---

> 📚 **Adaptación al español de la lección [Verification Gates]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).