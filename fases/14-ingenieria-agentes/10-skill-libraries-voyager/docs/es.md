# Skill libraries Voyager

> Voyager (Wang 2023): skill library para LLM agents. Skills = reusable code (functions, classes, modules, scripts) learned via curiosity-driven exploration. GPT-4 generates skills. Use tracking + LRU eviction. Composition: combine skills into pipeline. +Curriculum, +Code, +Reusable, +Composed, +Versioned, +Compatible, +Structured, +Multi-step. Skill types: (1) function (def foo), (2) class (class Foo), (3) module (import foo), (4) script (standalone), (5) composed (pipeline). Variants: Voyager (Wang 2023 +curriculum +code +GPT-4 gen +Minecraft), custom, smolagents (HF 2024 +code agents +lightweight +production), Anthropic Skills, OpenAI Agents, langchain. Frameworks: voyager, smolagents, langchain, autogen, anthropic. +Production: standard 2024-25. +Use cases: agent, RAG, code, automation, curriculum. Decision: curriculum -> Voyager, simple -> custom, code agents -> smolagents, production -> smolagents. Trade-offs: cada uno + specialty, custom + simple. 2025: +MCP + A2A + native + skills.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Skill class (name + code + description + uses).
- Implementar SkillLibrary con add/get/search/compose.
- Implementar LRU eviction.
- Implementar generate_skill_from_code.
- Diagnosticar Voyager vs custom vs smolagents.

## Constrúyelo

```python
class SkillLibrary:
    def add_skill(self, skill):
        if len(self.skills) >= self.max_size:
            lru = min(self.skills.values(), key=lambda s: (s.uses, -s.created_at))
            del self.skills[lru.name]
        self.skills[skill.name] = skill
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: voyager
fase: 14
leccion: 10
---

1. Skills = code.
2. Use tracking.
3. LRU eviction.
4. Composition.
5. +Reusable.
```

## Ejercicios

1. **Voyager**: implementar
   Voyager skill library.
2. **Smolagents**: probar
   smolagents code agents.
3. **Desafio**: custom
   skill library.

## Lecturas recomendadas

- "Voyager: An Open-Ended Embodied Agent with Large Language Models" (Wang et al., 2023)
- "Smolagents: Code Agents in Few Lines" (HF, 2024)
- "Anthropic Skills" (Anthropic, 2024)
- "OpenAI Agents SDK" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [Skill Libraries Voyager]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).