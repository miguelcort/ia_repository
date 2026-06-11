"""
Lección: 01-mdps-estados-acciones-y-recompensas
Fase: 09
MDP: tuple (S, A, P, R, gamma). Politica pi(a|s). Value functions.
"""
from __future__ import annotations
import sys
import numpy as np


def discounted_return(rewards, gamma, seed=None):
    """G_t = sum gamma^t * r_t.
    rewards: array de rewards por step. gamma: discount factor.
    """
    T = len(rewards)
    discounts = gamma ** np.arange(T)
    return float(np.sum(rewards * discounts))


def state_value_bellman(V_next, rewards, gamma):
    """V(s) = E[r + gamma * V(s')].
    Para sequences conocidas, V(s) = sum gamma^t * r_t.
    """
    return discounted_return(rewards, gamma)


def policy_evaluation(P, R, policy, gamma, n_states, n_actions, theta=1e-10):
    """Iterative policy evaluation.
    P: (S, A, S) -> prob. R: (S, A) -> reward. policy: (S) -> action.
    Returns: V (n_states,).
    """
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            a = policy[s]
            V[s] = sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(n_states))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(P, R, V, gamma, n_states, n_actions):
    """Greedy policy improvement dado V.
    policy_new(s) = argmax_a sum_s' P(s,a,s') * (R(s,a) + gamma * V(s'))
    """
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s2 in range(n_states):
                q_values[a] += P[s, a, s2] * (R[s, a] + gamma * V[s2])
        policy[s] = int(np.argmax(q_values))
    return policy


def sample_episode(policy, env_step, n_steps, seed=0):
    """Sample episodio de la politica en un env.
    env_step: (state) -> (next_state, reward, done).
    """
    rng = np.random.default_rng(seed)
    state = 0
    rewards = []
    for _ in range(n_steps):
        action = policy[state]
        # Mock: avanzar determinista
        next_state = min(state + 1, n_steps - 1)
        reward = 1.0 if next_state == n_steps - 1 else 0.0
        rewards.append(reward)
        if next_state == n_steps - 1:
            break
        state = next_state
    return rewards


def main() -> int:
    gamma = 0.9
    rewards = [0, 0, 0, 1]
    G = discounted_return(rewards, gamma)
    print(f"Discounted return: {G:.3f}")
    # Demo MDP: 3 estados, 2 acciones
    n_s, n_a = 3, 2
    P = np.zeros((n_s, n_a, n_s))
    P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5
    P[0, 1, 1] = 1.0
    P[1, 0, 0] = 0.5; P[1, 0, 2] = 0.5
    P[1, 1, 2] = 1.0
    P[2, :, 2] = 1.0
    R = np.array([[0, 0], [0, 0], [1, 1]])
    policy = np.array([0, 0, 0])
    V = policy_evaluation(P, R, policy, gamma, n_s, n_a)
    print(f"Value function: {V}")
    return 0


if __name__ == "__main__":
    sys.exit(main())