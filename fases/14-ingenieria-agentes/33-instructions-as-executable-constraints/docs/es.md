# Instructions as executable constraints

> Instructions as executable constraints: (1) Parse (MUST+MUST NOT+SHOULD), (2) Enforce (rules+code), (3) Validate (action+check), (4) Block (violation+callback), (5) MUST + MUST NOT (required+forbidden). Constraint: name+pattern+action+message, matches(text) search, check(text) bool+msg. ConstraintSet: constraints list+add(c)+validate(text) iterate+violations+block_on_violation(text, on_violation) callback. Parser: regex MUST NOT (.+?) + re.IGNORECASE -> action=block -> forbidden, regex MUST (.+?) + re.IGNORECASE -> action=require -> required, build Constraint list (pattern+action+message). Criterios: Constraints = explicit+parseable+runtime, Constitutional = principles+critique+self-edit, Llama Guard = classifier+13 cats+Meta. Decision: explicit rules -> constraints, self-critique -> Constitutional, multi-cat -> Llama Guard, mix -> all. Frameworks: anthropic, meta, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/32
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar Constraint con name + pattern + action + message.
- Implementar ConstraintSet con add + validate + block_on_violation.
- Implementar parse_instructions con regex MUST / MUST NOT.
- Diagnosticar enforcement.
- Diagnosticar constraints vs other.

## Constrúyelo

```python
class Constraint:
    def __init__(self, name, pattern, action, message=""):
        self.name = name
        self.pattern = re.compile(pattern)
        self.action = action
        self.message = message or f"Constraint {name} violated"

    def check(self, text):
        if self.pattern.search(text):
            return False, self.message
        return True, ""
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: executable-constraints
fase: 14
leccion: 33
---

1. Constraint + ConstraintSet.
2. parse_instructions.
3. validate + block.
4. +Production.
```

## Ejercicios

1. **Constraint**: probar
   pattern matching.
2. **parse**: probar
   MUST + MUST NOT.
3. **Desafio**: integrar
   con tu agent.

## Lecturas recomendadas

- "Anthropic: Claude Guardrails" (Anthropic, 2024)
- "Executable Specifications" (Cucumber, 2024)
- "RegEx-Based Validation" (NIST, 2024)

---

> 📚 **Adaptación al español de la lección [Instructions as Executable Constraints]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).