# 15 — Constitutional safety harness

> Constitutional safety harness: aplicar reglas (constitution) sobre outputs del LLM. Pipeline: generation → critique (LLM judge o rules engine) → revise (if fail) → output. Constitutional AI, LlamaGuard, custom rules.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 18 (safety), Fase 11
**Tiempo estimado:** 20 horas

## Objetivos

- Implementar critique-revise loop.
- Rules engine custom.
- LlamaGuard integration.
- Eval sobre HarmBench, AdvBench.

## El problema

Safety harness production: (1) Generation con LLM.
(2) Critique: LlamaGuard (Meta), custom classifier,
regex rules. (3) Revise: si falla, re-generar o
refuse. (4) Output: solo si pasa. Componentes:
LlamaGuard (multi-class), rules engine (regex +
keywords), LLM judge (Claude/GPT-4), refusal
classifier. Eval: HarmBench (250), AdvBench
(520), ToxicGen, BBQ. Métricas: ASR (attack
success rate), refusal rate, false positive rate.

## Constrúyelo

```python
def safety_harness(prompt, llm, llamaguard, rules):
    """Pipeline: generate -> critique -> revise -> output."""
    response = llm.generate(prompt)
    # LlamaGuard check
    if llamaguard.is_unsafe(response):
        return {"status": "refused",
                "reason": "llamaguard",
                "response": "I cannot help with that."}
    # Rules check
    if rules.violates(response):
        return {"status": "refused",
                "reason": "rules",
                "response": "I cannot help with that."}
    return {"status": "ok", "response": response}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-safety-harness
fase: 19
leccion: 15
---

1. LlamaGuard integration.
2. Custom rules engine.
3. Critique-revise loop.
4. Eval HarmBench.
5. False positive analysis.
```

## Ejercicios

1. **LlamaGuard**: configurar
   con custom policy.
2. **Rules engine**: 50 reglas
   de safety.
3. **Desafío**: ASR < 1% en
   HarmBench.

## Lecturas recomendadas

- "LlamaGuard 3" (Meta 2024)
- "HarmBench" (Mazeika 2024)
- "Constitutional AI" (Bai 2022)
- "guardrails-ai" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[15-constitutional-safety-harness]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
