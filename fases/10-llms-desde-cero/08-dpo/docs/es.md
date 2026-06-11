# DPO (Direct Preference Optimization)

> DPO (Rafailov 2023, Stanford): direct preference optimization sin critic, sin reward model. Closed-form: π(y|x) = π_ref(y|x) · exp(r/β) / Z. Loss: -log σ(β·(logπ_chosen - logπ_rejected - ref_chosen + ref_rejected)). Simple, +estable, +popular. Llama 3, Mistral-Instruct, Qwen 2 usan DPO. Variantes: IPO (regularized), KTO (binary, Kahneman-Tversky), ORPO (no reference), SimPO (length-normalized), R-DPO, CPO. Frameworks: TRL, axolotl, LLaMA-Factory. Iterative DPO (round 1, 2, 3) en producción.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/07-rlhf
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar DPO loss.
- Diagnosticar closed-form.
- Comparar DPO vs PPO.
- Implementar variantes (IPO, KTO).

## Constrúyelo

```python
def dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    chosen_diff = chosen_logp - ref_chosen_logp
    rejected_diff = rejected_logp - ref_rejected_logp
    margin = beta * (chosen_diff - rejected_diff)
    return -log(sigmoid(margin) + 1e-9)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dpo
fase: 10
leccion: 08
---

1. DPO: direct, no critic.
2. Closed-form: pi = pi_ref*exp(r/beta)/Z.
3. beta=0.1 tipico.
4. Variantes: IPO, KTO, ORPO, SimPO.
5. Llama 3, Mistral DPO.
```

## Ejercicios

1. **DPO**: implementar DPO en
   TRL con Llama 3 8B.
2. **KTO**: comparar DPO vs KTO
   en custom preferences.
3. **Desafio**: iterative DPO
   round 1, 2, 3.

## Lecturas recomendadas

- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- "KTO: Model Alignment as Prospect Theoretic Optimization" (Ethayarajh et al., 2024)
- "ORPO: Monolithic Preference Optimization without Reference Model" (Hong et al., 2024)
- "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[DPO]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).