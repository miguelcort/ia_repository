"""
Lección: 01-instruction-following-alignment-signal
Fase: 18
Prerrequisitos: 10-llms-desde-cero/06-instruction-tuning-sft,
                10-llms-desde-cero/07-rlhf,
                10-llms-desde-cero/08-dpo
Fuentes:
- Ouyang et al. — Training language models to follow instructions
  with human feedback (arXiv:2203.02155)
- Christiano et al. — Deep RL from human preferences (arXiv:1706.03741)
"""
from __future__ import annotations

import random
from collections.abc import Callable

ACTIONS = ("A", "B", "C")
TRUE_PREFERENCE = {"A": 0.1, "B": 0.7, "C": 0.2}  # P(labeler prefiere)


def softmax(logits: dict[str, float]) -> dict[str, float]:
    """Convierte logits en distribución de probabilidad."""
    m = max(logits.values())
    exp = {a: 2.718281828 ** (logits[a] - m) for a in logits}
    z = sum(exp.values())
    return {a: exp[a] / z for a in exp}


def labeler_sft(
    prompt: str, true_pref: dict[str, float] = TRUE_PREFERENCE
) -> str:
    """Etapa 1: el labeler responde siguiendo su distribución de
    preferencia."""
    r = random.random()
    acc = 0.0
    for action, p in true_pref.items():
        acc += p
        if r <= acc:
            return action
    return action


def fit_reward_model(
    pairwise: list[tuple[str, str, str]],  # (prompt, winner, loser)
    lr: float = 0.1,
    steps: int = 500,
) -> dict[str, float]:
    """Etapa 2: ajusta un reward model simple (un escalar por acción)
    con pérdida Bradley-Terry."""
    rewards = {a: 0.0 for a in ACTIONS}
    for _ in range(steps):
        prompt, winner, loser = random.choice(pairwise)
        r_w = rewards[winner]
        r_l = rewards[loser]
        # gradiente: derivada de -log sigmoid(r_w - r_l)
        # = -sigmoid(r_l - r_w) * d(r_w - r_l)
        scale = 1.0 / (1.0 + 2.718281828 ** (r_w - r_l))
        rewards[winner] += lr * scale
        rewards[loser] -= lr * scale
    return rewards


def bradley_terry_loss(rewards: dict[str, float]) -> float:
    """Calcula la pérdida promedio de Bradley-Terry sobre los pares
    almacenados en `rewards`. Útil para *tests*."""
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    total = 0.0
    for w, l in pairs:
        diff = rewards[w] - rewards[l]
        # -log sigmoid(diff) = log(1 + exp(-diff))
        total += (1.0 if diff > 0 else 2.718281828 ** (-diff))
    return total / len(pairs)


def generate_pairwise_data(
    n: int, true_pref: dict[str, float] = TRUE_PREFERENCE
) -> list[tuple[str, str, str]]:
    """Genera rankings por pares muestreando de la preferencia real."""
    data: list[tuple[str, str, str]] = []
    for i in range(n):
        # muestrea 4 acciones (con reemplazo) y dedupe preservando
        # preferencia para elegir winner y loser distintos
        sampled = random.choices(
            list(true_pref.keys()),
            weights=list(true_pref.values()),
            k=6,
        )
        unique = sorted(set(sampled), key=lambda a: -true_pref[a])
        winner = unique[0]
        # elige un loser con menor preferencia, garantizando diferencia
        losers = [a for a in unique if true_pref[a] < true_pref[winner]]
        if not losers:
            continue
        loser = random.choice(losers)
        data.append((f"prompt_{i}", winner, loser))
    return data


def ppo_update(
    policy_logits: dict[str, float],
    sft_logits: dict[str, float],
    rewards: dict[str, float],
    beta: float,
    lr: float = 0.05,
) -> dict[str, float]:
    """Etapa 3: un paso PPO simplificado con penalización KL.
    Devuelve nuevos logits de política."""
    pi = softmax(policy_logits)
    pi_sft = softmax(sft_logits)
    new_logits = dict(policy_logits)
    for a in ACTIONS:
        # gradiente surrogate: ventaja - beta * KL gradient
        advantage = rewards[a] - sum(pi[b] * rewards[b] for b in ACTIONS)
        kl_grad = pi[a] - pi_sft[a]
        new_logits[a] += lr * (advantage - beta * kl_grad)
    return new_logits


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q)."""
    total = 0.0
    for a in p:
        if p[a] > 0:
            total += p[a] * (2.718281828 ** (p[a] - q[a]) - 1)
    return total


def main() -> int:
    """Demo del pipeline InstructGPT en 3 etapas."""
    random.seed(42)

    print("=== 01-instruction-following-alignment-signal ===")
    print()

    # Etapa 1: SFT — la política SFT imita la preferencia del labeler
    sft_counts: dict[str, int] = {a: 0 for a in ACTIONS}
    for i in range(200):
        action = labeler_sft(f"prompt_{i}")
        sft_counts[action] += 1
    sft_logits = {a: sft_counts[a] / 200.0 for a in ACTIONS}
    sft_probs = softmax(sft_logits)
    print(f"SFT (n=200): {sft_probs}")

    # Etapa 2: RM — ajusta el reward model con datos por pares
    pairwise = generate_pairwise_data(500)
    rewards = fit_reward_model(pairwise)
    print(f"Recompensas RM: {rewards}")
    print(f"Pérdida BT final: {bradley_terry_loss(rewards):.4f}")
    print()

    # Etapa 3: PPO con KL penalty
    print("Etapa 3 — PPO con KL penalty:")
    for beta in (0.1, 0.0):
        random.seed(7)
        policy = dict(sft_logits)
        for step in range(50):
            policy = ppo_update(policy, sft_logits, rewards, beta=beta)
            if step in (0, 10, 49):
                kl = kl_divergence(softmax(policy), sft_probs)
                print(
                    f"  beta={beta:.1f}  step={step:>3}  "
                    f"pi={softmax(policy)}  KL(pi||pi_SFT)={kl:.4f}"
                )
        print()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
