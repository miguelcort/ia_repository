# Role specialization

> Role specialization: (1) Prompts (system+persona), (2) Tools (per role+subset), (3) Constraints (temperature+behavior), (4) Templates (reusable+dev+test+review+research+PM). ROLE_TEMPLATES: developer (code+test+tools: read+write+edit+bash+search+temp 0.2), tester (QA+tools: read+bash+test-runner+temp 0.1), reviewer (audit+tools: read+grep+linter+temp 0.0), researcher (search+tools: search+read+fetch+temp 0.3), product_manager (PM+tools: read+write+temp 0.5). SpecialistAgent: role_name+system_prompt+tools+temperature+can_use(tool)+describe(). Ventajas: clarity (clear role+persona), performance (focused+better quality), tool subset (relevant+no noise), audit (clear responsibility+trace), reuse (templates+composable). Criterios: Roles = distinct+audit+specialized, Shared = simple+fast+generic, Few-shot = examples+pattern+in-context. Decision: distinct -> roles, simple -> shared, examples -> few-shot, mix -> roles+few-shot. Frameworks: langchain, crewai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + roles.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ROLE_TEMPLATES con 5 roles.
- Implementar SpecialistAgent con role_name + system_prompt + tools + temperature.
- Implementar can_use(tool) + describe().
- Diagnosticar templates.
- Diagnosticar roles vs shared vs few-shot.

## Constrúyelo

```python
class SpecialistAgent:
    def __init__(self, role_name, custom_prompt=None):
        self.role_name = role_name
        template = ROLE_TEMPLATES.get(role_name)
        if not template:
            raise ValueError(f"unknown role: {role_name}")
        self.system_prompt = custom_prompt or template["system_prompt"]
        self.tools = list(template["tools"])
        self.temperature = template["temperature"]
        self.agent_id = str(uuid.uuid4())
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: role-specialization
fase: 16
leccion: 08
---

1. ROLE_TEMPLATES.
2. SpecialistAgent.
3. can_use + describe.
4. +Production.
```

## Ejercicios

1. **Templates**: probar
   5 roles.
2. **SpecialistAgent**: probar
   can_use.
3. **Desafio**: integrar
   con CrewAI roles.

## Lecturas recomendadas

- "CrewAI: Roles" (CrewAI, 2024)
- "Anthropic: System Prompts" (Anthropic, 2024)
- "OpenAI: Custom GPT" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [Role Specialization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).