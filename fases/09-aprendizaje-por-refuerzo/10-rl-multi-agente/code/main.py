"""
Lección: 10-rl-multi-agente
Fase: 09
MARL: Multi-agent RL. Stochastic games, Nash equilibrium.
Cooperative, competitive, mixed.
"""
from __future__ import annotations
import sys
import numpy as np


def stochastic_game_step(state, action_a, action_b, P, R_a, R_b, n_states):
    """Sample next state de stochastic game.
    Returns: next_state, reward_a, reward_b.
    """
    probs = P[state, action_a, action_b]
    next_state = int(np.random.choice(n_states, p=probs))
    r_a = R_a[state, action_a, action_b]
    r_b = R_b[state, action_a, action_b]
    return next_state, r_a, r_b


def nash_equilibrium_2p_payoff(payoff_a, payoff_b):
    """Encuentra Nash equilibrium en 2-player game.
    Simplificado: pure strategy Nash.
    """
    n_a, n_b = payoff_a.shape
    # Best response para cada action de B
    equilibria = []
    for a in range(n_a):
        for b in range(n_b):
            # A: a es best response a b?
            a_best = np.argmax(payoff_a[:, b])
            b_best = np.argmax(payoff_b[a, :])
            if a == a_best and b == b_best:
                equilibria.append((a, b))
    return equilibria


def joint_action_to_idx(a, b, n_b):
    """Convertir (a, b) a joint action index."""
    return a * n_b + b


def q_marl_update(Q, s, a, b, r_a, r_b, s_next, alpha, gamma, n_actions):
    """Q-learning para 2 agentes.
    Q_a(s, a, b) <- Q_a(s, a, b) + alpha * (r_a + gamma * max Q_a(s', a', b') - Q_a(s, a, b)).
    """
    target = r_a + gamma * np.max(Q[s_next, :, :])
    Q[s, a, b] = Q[s, a, b] + alpha * (target - Q[s, a, b])
    return Q


def coop_q_update(Q, s, joint_action, r, s_next, alpha, gamma, n_joint_actions):
    """Q-learning para team: single shared Q.
    """
    target = r + gamma * np.max(Q[s_next])
    Q[s, joint_action] = Q[s, joint_action] + alpha * (target - Q[s, joint_action])
    return Q


def centralised_vs_decentralised():
    """Comparacion."""
    return {
        "Centralised": "Un agente controla todos. Simple, no coordination.",
        "Decentralised": "Cada agente independiente. Communication cost.",
        "Centralised training, decentralised exec (CTDE)": "Train con global info, exec con local.",
        "MADDPG, QMIX, MAPPO": "Algoritmos SOTA MARL con CTDE.",
        "StarCraft, Diplomacy, Hanabi": "Benchmarks MARL.",
    }


def main() -> int:
    # Prisoner's dilemma simple
    # Payoff matrix: (A, B) -> R_a, R_b
    R_a = np.array([
        [(3, 3)],  # (cooperate, cooperate): ambos 3
        [(0, 5)],  # (defect, cooperate): A 0, B 5
    ]).squeeze()  # (2, 2) ahora
    # Real: R_a (2, 2), R_b (2, 2)
    R_a = np.array([[3, 0], [5, 1]])
    R_b = np.array([[3, 5], [0, 1]])
    eq = nash_equilibrium_2p_payoff(R_a, R_b)
    print(f"Nash equilibria: {eq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())