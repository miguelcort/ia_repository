# Automated alignment research

> Automated alignment research: AI agents researching AI safety. (1) Red-teaming (generate attacks + test target + measure ASR attack success rate, +Safety +Detection +ASR), (2) Self-play, (3) Debate (multiple agents argue about safety + round-robin + consensus, +Safety +Robust +Diverse), (4) Safety eval (dataset + measure safe rate, +Standard +Safe rate), (5) Custom. +Safety, +Alignment, +Production, +Reliable, +Standard, +Robust, +Diverse, +Detection, +Reliable, +Consensus. Variants: red-teaming, debate, safety eval, self-play, custom, Llama Guard, Anthropic RSP. Frameworks: anthropic, openai, deepmind, custom, apollo. +Production: standard 2024-25. +Use cases: safety, alignment, red-team, eval, attack. Decision: detect -> red-team, consensus -> debate, measure -> safety eval, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + alignment.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar automated_red_team con ASR.
- Implementar automated_debate_safety con consensus.
- Implementar automated_safety_eval.
- Diagnosticar red-teaming vs debate vs safety eval.

## Constrúyelo

```python
def automated_red_team(target_llm, n_attempts=10, attack_fn=None):
    attack_fn = attack_fn or (lambda i: f"attack_{i}")
    successes = 0
    attempts = []
    for i in range(n_attempts):
        attack = attack_fn(i)
        is_success = i % 4 == 0
        if is_success:
            successes += 1
        attempts.append({"attack": attack, "success": is_success})
    return {"n_attempts": n_attempts, "successes": successes, "asr": successes / n_attempts, "attempts": attempts}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: alignment-research
fase: 15
leccion: 06
---

1. Red-teaming.
2. Debate.
3. Safety eval.
4. ASR + consensus.
5. +Safety.
```

## Ejercicios

1. **Red-teaming**:
   implementar red team custom.
2. **Debate**: probar
   safety debate.
3. **Desafio**: full
   alignment pipeline.

## Lecturas recomendadas

- "Automated Red-Teaming with LLMs" (Anthropic, 2024)
- "Constitutional AI" (Anthropic, 2022)
- "Red-Teaming Language Models" (Perez et al., 2022)
- "AI Safety via Debate" (Irving et al., OpenAI, 2018)

---

> 📚 **Adaptación al español de la lección [Automated Alignment Research]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).