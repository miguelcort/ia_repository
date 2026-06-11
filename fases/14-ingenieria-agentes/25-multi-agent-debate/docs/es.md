# Multi-agent debate

> Multi-agent debate (Du et al. 2023, "Improving Factuality via Multi-Agent Debate"): multiple agents argue + judge decides + +factuality +reasoning +production. Components: (1) Debater (name + stance + model + arguments list), (2) Judge (name + model + decide based on arguments). Loop: (1) round: cada agent argues (incorporate other_arguments prefix "Counter to N args:"), (2) counter-arguments, (3) judge decide winner. Beneficios: +Factuality (multiple perspectives +truth), +Reasoning (counter-arguments +logic), +Quality (diverse views +refined), -Hallucinations (cross-check), -Bias (multiple stances), +Production. Variants: Du 2023 seminal, Society of Mind, LLM-Debate, custom. Frameworks: langchain, openai, custom, autogen, crewai, smolagents. +Production: standard 2024-25. +Use cases: agent, multi-agent, debate, factuality, reasoning, fact-checking. Decision: factuality -> debate, simple -> single, quality -> Self-Refine, production -> debate + self-refine. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + debate.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13, 14/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Debater con arguments list.
- Implementar Judge con decide.
- Implementar multi_agent_debate loop.
- Diagnosticar debate vs single vs Self-Refine.
- Diagnosticar beneficios.

## Constrúyelo

```python
class Debater:
    def argue(self, topic, other_arguments=None):
        prefix = ""
        if other_arguments:
            prefix = f"Counter to {len(other_arguments)} args: "
        return f"{prefix}[{self.name}] argues about '{topic[:30]}'"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: debate
fase: 14
leccion: 25
---

1. Debater + argue.
2. Judge + decide.
3. Debate loop.
4. Counter-arguments.
5. +Factuality.
```

## Ejercicios

1. **Debate**: implementar
   multi-agent debate con
   LLM real.
2. **Judge**: probar
   judge LLM-based.
3. **Desafio**: debate
   production con factuality.

## Lecturas recomendadas

- "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (Du et al., 2023)
- "Society of Mind" (Minsky, 1986)
- "Multi-Agent Debate Best Practices" (Anthropic, 2024)
- "LLM-Debate" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [Multi-Agent Debate]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).