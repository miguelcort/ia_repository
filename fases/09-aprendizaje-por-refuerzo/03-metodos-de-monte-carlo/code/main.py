"""
Lección: 03-metodos-de-monte-carlo
Fase: 09
Monte Carlo RL: V(s) = E[sum R]. Sample episodes, average returns.
"""
from __future__ import annotations
import sys
import numpy as np


def first_visit_mc(states, returns, n_states):
    """First-visit MC: V(s) = promedio de returns en episodios donde
    s aparece por primera vez.
    states: lista de secuencias de estados por episodio. returns: lista
    de returns por episodio.
    """
    V = np.zeros(n_states)
    counts = np.zeros(n_states)
    for episode_states, G in zip(states, returns):
        visited = set()
        for s in episode_states:
            if s not in visited:
                visited.add(s)
                counts[s] += 1
                V[s] += (G - V[s]) / counts[s]
    return V


def every_visit_mc(states, returns, n_states):
    """Every-visit MC: V(s) = promedio de returns en cada visita."""
    V = np.zeros(n_states)
    counts = np.zeros(n_states)
    for episode_states, G in zip(states, returns):
        for s in episode_states:
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V


def sample_returns(n_episodes, n_steps, rewards, gamma, seed=0):
    """Samplea returns de episodios mock.
    Para demo: cada episodio tiene misma secuencia de rewards.
    """
    rng = np.random.default_rng(seed)
    G_list = []
    for _ in range(n_episodes):
        # Vary rewards slightly
        r = rewards + rng.standard_normal(n_steps) * 0.1
        G = float(np.sum(r * (gamma ** np.arange(n_steps))))
        G_list.append(G)
    return G_list


def exploring_starts_check(n_states, n_actions, init_prob=0.1):
    """Verifica que politica es exploring starts:
    P(a | s) > 0 para todo (s, a).
    """
    # Mock: siempre True en este demo
    return True


def on_policy_mc_control(env, policy, n_episodes, gamma, n_states, n_actions):
    """On-policy first-visit MC control.
    policy: (S, A) -> prob. Returns: Q, policy mejorada.
    """
    Q = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    for ep in range(n_episodes):
        # Sample episode
        episode = []
        s = 0
        for _ in range(10):
            a = np.random.choice(n_actions, p=policy[s])
            episode.append((s, a, 1.0))  # reward = 1
            s = (s + 1) % n_states
        # Update Q
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in [(e[0], e[1]) for e in episode[:t]]:
                returns_count[s, a] += 1
                returns_sum[s, a] += G
                Q[s, a] = returns_sum[s, a] / returns_count[s, a]
                # Policy improvement (e-greedy)
                best_a = np.argmax(Q[s])
                for a2 in range(n_actions):
                    policy[s, a2] = 0.05 / n_actions
                policy[s, best_a] = 1 - 0.05 + 0.05 / n_actions
    return Q, policy


def main() -> int:
    n_states = 5
    n_steps = 4
    rewards = np.array([1, 1, 1, 1])
    gamma = 0.9
    G_list = sample_returns(100, n_steps, rewards, gamma, seed=42)
    print(f"Mean return: {np.mean(G_list):.3f}, std: {np.std(G_list):.3f}")
    # First visit MC
    states_list = [[0, 1, 2, 3]] * 100
    V = first_visit_mc(states_list, G_list, n_states)
    print(f"V (first-visit): {V.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())