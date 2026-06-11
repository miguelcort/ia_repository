# RLHF (Reinforcement Learning from Human Feedback)

> RLHF para LLMs: pipeline SFT → Reward Model (Bradley-Terry con preferences humanas) → PPO con reference model y KL penalty. Política = SFT + LoRA, reference = SFT frozen, KL(π || π_ref) × β (β=0.05-0.1). Reward hacking mitigations: KL, iterative RLHF, ensemble RMs, Constitutional AI, RLAIF. Variantes: PPO (Llama 2), DPO (Llama 3, Mistral), GRPO (DeepSeek), KTO, RLAIF, Constitutional AI. Frameworks: TRL (HuggingFace), OpenRLHF, DeepSpeed-Chat.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/09-reward-modeling-y-rlhf, 10/06-instruction-tuning-sft
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Bradley-Terry loss.
- Diagnosticar pipeline SFT -> RM -> PPO.
- Diagnosticar reward hacking risks.
- Comparar variantes RLHF.

## Constrúyelo

```python
def bradley_terry_loss(chosen, rejected):
    return -log(sigmoid(chosen - rejected))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rlhf-llm
fase: 10
leccion: 07
---

1. SFT -> RM -> PPO.
2. KL penalty beta=0.05-0.1.
3. Reward hacking mitigation.
4. DPO, GRPO variants.
5. TRL, OpenRLHF frameworks.
```

## Ejercicios

1. **PPO**: implementar RLHF con
   TRL en Llama 3 8B.
2. **Iterative**: round 1, 2, 3 con
   nuevos preferences.
3. **Desafio**: Constitutional AI
   simple.

## Lecturas recomendadas

- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
- "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- "TRL: Transformer Reinforcement Learning" (HuggingFace, 2023)

---

> 📚 **Adaptación al español** de la lección "[RLHF]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).