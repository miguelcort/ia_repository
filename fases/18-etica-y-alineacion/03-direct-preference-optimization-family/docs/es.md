# 03 — DPO family: DPO, IPO, KTO, ORPO, SimPO

> DPO (Rafailov 2023) reformula RLHF como supervised learning: el ratio π_θ/π_ref con KL implícito. IPO, KTO, ORPO, SimPO son variantes que arreglan problemas específicos de DPO (overfit, length bias, format dependence).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/08-dpo
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar DPO loss.
- Implementar IPO, KTO, ORPO, SimPO.
- Comparar comportamiento en datos sintéticos.
- Diagnosticar cuándo usar cada variante.

## El problema

DPO (Rafailov 2023, Stanford) mostró que RLHF es
equivalente a una pérdida supervisada sobre preferencias,
sin reward model ni PPO. Pero DPO tiene problemas:
overfit a datos de preferencia, length bias, y
difficulty para datos con quality scores vs preferences.
La familia creció: IPO (identity preference, sin
overfit), KTO (matching, binary feedback), ORPO
(odds ratio, sin reference model), SimPO (simple
preference optimization, length-normalized).

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_prob(logits, target):
    """Log softmax + gather."""
    log_softmax = logits - np.log(np.exp(logits).sum(-1)
                                  + 1e-8)
    return log_softmax[target]


def dpo_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1):
    """L_DPO = -log σ(β log(π/π_ref)(chosen - rejected))."""
    diff = beta * ((policy_chosen - ref_chosen)
                  - (policy_rejected - ref_rejected))
    return -np.log(sigmoid(diff) + 1e-8).mean()


def ipo_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1, tau=0.1):
    """IPO: (log ratio diff / (1/beta))^2 / 2 + tau.
    Sin overfit, sin KL collapse."""
    diff = (policy_chosen - ref_chosen
           - (policy_rejected - ref_rejected)) / (2 * beta)
    return (diff ** 2).mean() / 2 + tau


def kto_loss(policy_chosen, policy_rejected, ref_chosen,
            ref_rejected, beta=0.1, desirable=True):
    """KTO: matching, acepta binary feedback (good/bad).
    Asimétrico: desirable vs undesirable."""
    kl_chosen = beta * (policy_chosen - ref_chosen)
    kl_rejected = beta * (policy_rejected - ref_rejected)
    if desirable:
        return -np.log(sigmoid(kl_chosen) + 1e-8).mean()
    else:
        return -np.log(1 - sigmoid(kl_rejected) + 1e-8).mean()


def orpo_loss(policy_chosen, policy_rejected,
            sft_chosen, sft_rejected, beta=0.1):
    """ORPO: odds ratio. Sin reference model, SFT + ORPO loss.
    log σ(log odds(π/SFT) - log odds(π/SFT)|rejected)."""
    log_odds_chosen = (policy_chosen - sft_chosen)
    log_odds_rejected = (policy_rejected - sft_rejected)
    return -np.log(sigmoid(beta * (log_odds_chosen
                                  - log_odds_rejected))
                  + 1e-8).mean()


def simpo_loss(policy_chosen, policy_rejected, beta=2.0,
            gamma=1.0):
    """SimPO: length-normalized preference optimization.
    Sin reference model. L = -log σ(β/|y| Σ log π - γ)."""
    # Normalize by length (assumed passed as mean over tokens)
    return -np.log(sigmoid(beta * policy_chosen - gamma
                          - (beta * policy_rejected - gamma))
                  + 1e-8).mean()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dpo-family
fase: 18
leccion: 03
---

1. DPO: default, requiere ref model.
2. IPO: sin overfit, datos ruidosos.
3. KTO: binary feedback (thumbs up/down).
4. ORPO: SFT + alignment en un solo paso.
5. SimPO: length-normalized, sin ref model.
6. Comparar en datos propios.
```

## Ejercicios

1. **DPO**: implementar y entrenar en
   UltraFeedback.
2. **IPO vs DPO**: comparar overfit en datos
   ruidosos.
3. **SimPO**: usar en QA, comparar con DPO.
4. **Desafío**: entrenar ORPO en Mixtral 8x7B
   instruct.

## Cuándo usar cada variante

DPO (Rafailov 2023): default, requiere ref model
(pi_ref). Bueno para chat general, summarization.
β=0.1-0.5. Limitaciones: (1) overfit a preferencias.
(2) length bias. (3) necesita paired data.

IPO (Azar 2023): identity preference optimization.
Para datos ruidosos, weak preferences. Sin
overfit por construcción. β=0.1, τ=0.1.

KTO (Ethayarajh 2024): Kahneman-Tversky
optimization. Acepta binary feedback (chosen vs
not chosen), no necesita pares. Bueno para
production data (thumbs up/down). β=0.1.

ORPO (Hong 2024): odds ratio preference
optimization. SFT + alignment en un solo paso.
Sin reference model. Más eficiente. β=0.1-0.5.

SimPO (Meng 2024): simple preference
optimization. Length-normalized log probs.
Sin reference model. Bueno para long-context.
β=2.0, γ=1.0-2.0.

RLAIF (Bai 2022): Constitutional AI. LLM judge
en vez de humanos. Scale preference data. DPO
después.

Frameworks: TRL (Hugging Face, DPO, IPO, KTO, ORPO).
TRL incluye reference model auto-cargado. axolotl
y LLaMA-Factory soportan varios.

## Trampas

DPO overfit: si datos de preferencia son ruidosos
o small, IPO es mejor. KTO para binary feedback.
ORPO cuando no quieres ref model. SimPO para
length bias.

Length bias: SimPO normaliza por length, mejor
para long answers. DPO tiende a preferir longer.

## Lecturas recomendadas

- "Direct Preference Optimization" (Rafailov 2023)
- "A General Theoretical Paradigm to Understand
  Learning from Human Feedback" (Azar 2023, IPO)
- "KTO: Model Alignment as Prospect Theoretic
  Optimization" (Ethayarajh 2024)
- "ORPO: Monolithic Preference Optimization
  without Reference Model" (Hong 2024)
- "SimPO: Simple Preference Optimization with a
  Reference-Free Reward" (Meng 2024)
- HuggingFace TRL: https://huggingface.co/docs/trl

---

> 📚 **Adaptación al español** de la lección
> "[03-direct-preference-optimization-family]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
