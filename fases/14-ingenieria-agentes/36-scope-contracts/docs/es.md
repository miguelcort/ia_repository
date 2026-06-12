# Scope contracts

> Scope contracts: (1) In-scope (allowed+read/write), (2) Out-of-scope (forbidden+delete), (3) Validate actions (check+allow/deny), (4) Prevent scope creep (strict+audit), (5) Formal contracts (parse+register), (6) Multi-tier (dev/prod). ScopeContract: name+in_scope (list+lower)+out_of_scope (list+lower)+allows(action) lower+check out first+check in+not_in_scope+reason. ScopeRegistry: contracts list+add(c)+validate(action) iterate+first match+no_contract_allows. parse_scope: re.split [,;\n] + strip + build in_scope + out_of_scope + return ScopeContract. Criterios: Scope = action-level+allowed/forbidden+tier, Constraints = pattern-level+regex+runtime, Reviewer = post-hoc+quality+audit. Decision: action -> scope, pattern -> constraints, quality -> reviewer, mix -> all. Frameworks: langchain, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + scope.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/35
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ScopeContract con in_scope + out_of_scope + allows.
- Implementar ScopeRegistry con add + validate.
- Implementar parse_scope desde texto libre.
- Diagnosticar in/out scope.
- Diagnosticar scope vs other.

## Constrúyelo

```python
class ScopeContract:
    def allows(self, action):
        action_lower = action.lower()
        for forbidden in self.out_of_scope:
            if forbidden in action_lower:
                return False, f"out_of_scope: {forbidden}"
        for allowed in self.in_scope:
            if allowed in action_lower:
                return True, "in_scope"
        return False, "not_in_scope"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: scope-contracts
fase: 14
leccion: 36
---

1. ScopeContract.
2. ScopeRegistry.
3. parse_scope.
4. +Production.
```

## Ejercicios

1. **ScopeContract**: probar
   in_scope + out_of_scope.
2. **ScopeRegistry**: probar
   multi-tier.
3. **Desafio**: integrar
   con tu agent.

## Lecturas recomendadas

- "Design by Contract" (Meyer, 1986)
- "Scope Management" (Anthropic, 2024)
- "Erlang: gen_server" (Erlang, 2024)

---

> 📚 **Adaptación al español de la lección [Scope Contracts]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).