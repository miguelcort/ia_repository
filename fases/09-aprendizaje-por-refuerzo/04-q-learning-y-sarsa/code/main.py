"""
Lección: 04-q-learning-y-sarsa
Fase: 09
Temporal Difference (TD) learning. Q-learning (off-policy) y SARSA (on-policy).
"""
from __future__ import annotations
import sys
import numpy as np


def q_learning_update(Q, s, a, r, s_next, alpha, gamma, n_actions):
    """Q-learning: Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)).
    Off-policy (target usa max, no action tomada).
    """
    target = r + gamma * np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma):
    """SARSA: Q(s, a) = Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a)).
    On-policy (usa a' real taken).
    """
    target = r + gamma * Q[s_next, a_next]
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def expected_sarsa_update(Q, s, a, r, s_next, alpha, gamma, policy):
    """Expected SARSA: usa expectation over policy.
    Q(s, a) = Q(s, a) + alpha * (r + gamma * sum_a' pi(a'|s') * Q(s', a') - Q(s, a)).
    """
    expected_q = sum(p * Q[s_next, a2] for a2, p in enumerate(policy[s_next]))
    target = r + gamma * expected_q
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def epsilon_greedy(Q, s, n_actions, epsilon):
    """Epsilon-greedy action selection."""
    rng = np.random.default_rng(0)
    if rng.uniform() < epsilon:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(Q[s]))


def q_learning_episode(Q, env_step, alpha, gamma, epsilon, n_actions, n_steps):
    """Run un episodio de Q-learning."""
    s = 0
    rewards = 0
    for t in range(n_steps):
        a = epsilon_greedy(Q, s, n_actions, epsilon)
        s_next, r, done = env_step(s, a)
        Q = q_learning_update(Q, s, a, r, s_next, alpha, gamma, n_actions)
        rewards += r
        if done:
            break
        s = s_next
    return Q, rewards


def td_lambda_return(returns, td_errors, lambda_):
    """TD(lambda) return: G_t^lambda = (1-lambda) * sum_{n=1}^inf lambda^{n-1} G_t^{(n)}.
    Equivalente: G_t^lambda = sum_{n=1}^inf lambda^{n-1} * delta_t * sum_{i=1}^n gamma^{i-1}.
    Mock: linear combination de returns y TD errors.
    """
    T = len(returns)
    G = [0.0] * T
    for t in range(T):
        G_t = 0
        for n in range(1, T - t):
            n_step = sum(returns[t:t + n])
            G_t += (lambda_ ** (n - 1)) * n_step
        G[t] = (1 - lambda_) * G_t
    return G


def main() -> int:
    n_states, n_actions = 5, 4
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    # Mock env: state -> state+1, reward = 1 al final
    def env_step(s, a):
        s_next = min(s + 1, n_states - 1)
        r = 1.0 if s_next == n_states - 1 else 0.0
        done = s_next == n_states - 1
        return s_next, r, done
    for ep in range(200):
        Q, _ = q_learning_episode(Q, env_step, alpha, gamma, epsilon, n_actions, 20)
    print(f"Q final: {Q.round(2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())