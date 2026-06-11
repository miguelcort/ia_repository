# Reward modeling y RLHF

> RLHF: pipeline de 3 pasos: SFT (instrucciones) → Reward Model (Bradley-Terry con preferences humanas) → PPO con KL constraint. DPO (Rafailov 2023): direct preference optimization sin critic ni reward model. Variantes: KTO (Kahneman-Tversky), GRPO (group baseline, DeepSeek-R1), IPO, RLAIF, Constitutional AI. Reward hacking: KL penalty, ensemble RMs, iterative. Hoy: PPO default, DPO gaining popularity.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/08-ppo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Bradley-Terry loss.
- Calcular pairwise accuracy.
- Implementar DPO loss.
- Implementar GRPO loss simple.
- Diagnosticar reward hacking.

## Constrúyelo

```python
def bradley_terry_loss(chosen_reward, rejected_reward):
    return -np.log(sigmoid(chosen_reward - rejected_reward) + 1e-9)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rlhf
fase: 09
leccion: 09
---

1. SFT -> RM -> PPO.
2. BT loss con preferences.
3. KL constraint beta=0.05-0.1.
4. DPO, GRPO, KTO.
5. Reward hacking mitigations.
```

## Ejercicios

1. **RM**: entrenar RM con
   preferences sinteticas.
2. **DPO**: implementar DPO en
   TRL.
3. **Desafio**: Constitutional AI
   simple.

## Lecturas recomendadas

- "Deep Reinforcement Learning from Human Preferences" (Christiano et al., 2017)
- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Reward Modeling RLHF]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).