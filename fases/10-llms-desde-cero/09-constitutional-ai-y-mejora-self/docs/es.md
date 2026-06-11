# Constitutional AI y mejora self

> Constitutional AI (Bai 2022, Anthropic): set de principios escritos, self-critique contra constitution, AI genera revisión, train en (prompt, original, critique, revision). RLAIF (AI feedback en lugar de human): más barato, escalable. Self-Rewarding LLM (Meta 2024): LLM-as-judge + actor. Aplicaciones: Claude 2/3, Spark, otros. Riesgos: reward hacking, mode collapse, bias amplification, distribution shift. Mitigations: human eval periódico, KL constraint, ensemble RMs.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/07-rlhf, 10/08-dpo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar constitutional principles.
- Diagnosticar RLAIF pipeline.
- Comparar Constitutional AI vs RLHF.
- Diagnosticar riesgos de self-improvement.

## Constrúyelo

```python
def self_critique_score(response, principles):
    rng = np.random.default_rng(0)
    violations = sum(rng.uniform() < 0.3 for _ in principles)
    return violations / len(principles)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-constitutional-ai
fase: 10
leccion: 09
---

1. Constitution: principles.
2. Self-critique + revision.
3. RLAIF: AI feedback.
4. Self-Rewarding LLM.
5. Anthropic Claude usa Constitutional AI.
```

## Ejercicios

1. **Constitution**: escribir
   constitution para dominio custom.
2. **RLAIF**: pipeline con AI judge.
3. **Desafio**: self-rewarding
   LLM simple.

## Lecturas recomendadas

- "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
- "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Lee et al., 2023)
- "Self-Rewarding Language Models" (Yuan et al., 2024)
- "Specific versus General Principles for Constitutional AI" (Anthropic, 2023)

---

> 📚 **Adaptación al español** de la lección "[Constitutional AI Self Improvement]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).