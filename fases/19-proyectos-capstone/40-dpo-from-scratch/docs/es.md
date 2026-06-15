# 40 — DPO from scratch

> DPO desde cero: pérdida directa sobre (prompt, chosen, rejected). Sin RM, sin PPO. Implementación: Hugging Face TRL DPOTrainer o custom PyTorch loop. Datasets: UltraFeedback, HelpSteer, Argilla.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/39, Fase 18/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar DPO loss.
- DPOTrainer (TRL).
- Eval preference accuracy.
- Compare con PPO.

## Constrúyelo

```python
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
           ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """L_DPO = -log σ(β log(π/π_ref)(chosen - rejected))."""
    diff = beta * ((policy_chosen_logps - ref_chosen_logps)
                  - (policy_rejected_logps - ref_rejected_logps))
    return -F.logsigmoid(diff).mean()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dpo-scratch
fase: 19
leccion: 40
---

1. DPO loss.
2. DPOTrainer.
3. UltraFeedback.
4. Preference acc.
5. Compare PPO.
```

## Ejercicios

1. **DPO**: 10K pairs
   UltraFeedback.
2. **DPO loss**: custom
   impl.
3. **Desafío**: +
   alignment eval.

## Detalles

DPO loss derivation: derivada de RLHF. Mismo
equilibrio entre política y reference, pero como
classification loss sobre pares. Implementation: solo
necesitas log π_θ(y_w|x) y log π_ref(y_w|x), chosen y
rejected. Sin reward model, sin PPO loop, sin
advantage estimation.

TRL DPOTrainer: DPOConfig con beta=0.1, rpo_alpha=None,
loss_type="sigmoid" (DPO) o "ipo" o "kto" o "simpo".
Reference model cargado en memoria (4x memoria vs
SFT). Reference-free variants: ORPO, SimPO (no ref
model needed).

DPO datasets: UltraFeedback (Cui 2023, 64K pairs,
GPT-4 judge), HelpSteer (NVIDIA, 37K, multi-attr),
Argilla DPO mix, OpenAssistant, StackExchange
preferences.

Hyperparam tuning: beta crítico. beta=0.1 default.
beta bajo (0.01): drift libre. beta alto (1.0):
estancado cerca SFT. Grid search.

DPO issues: (1) Overfit si pocos pares. (2) Length
bias: prefiere longer. (3) Verbosity bias. Mitigations:
SimPO (length-norm), ORPO (no ref), KTO (binary fb).

Hoy: DPO + UltraFeedback + TRL es el standard para
chat models. Llama 3 Instruct, Qwen 2.5 Chat, Mistral
Instruct todos usan variantes DPO/IPO/KTO.

## Lecturas recomendadas

- "DPO" (Rafailov 2023)
- "UltraFeedback" (Cui 2023)
- "TRL DPOTrainer" (2024)
- "Argilla DPO" (2024)
- "SimPO" (Meng 2024)

---

> 📚 **Adaptación al español** de la lección
> "[40-dpo-from-scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
