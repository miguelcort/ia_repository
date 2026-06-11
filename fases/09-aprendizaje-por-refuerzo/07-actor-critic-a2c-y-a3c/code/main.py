"""
Lección: 07-actor-critic-a2c-y-a3c
Fase: 09
Actor-Critic: policy (actor) + value (critic). A2C (Advantage Actor-Critic),
A3C (Asynchronous).
"""
from __future__ import annotations
import sys
import numpy as np
import math


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ActorCritic:
    """Actor-Critic con policy (actor) y value (critic) heads."""

    def __init__(self, n_states, n_actions, hidden=32, seed=0, gamma=0.99, lr=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_states)
        # Shared
        self.W_shared = s * rng.standard_normal((n_states, hidden))
        self.b_shared = np.zeros(hidden)
        # Actor head
        self.W_actor = s * rng.standard_normal((hidden, n_actions))
        self.b_actor = np.zeros(n_actions)
        # Critic head
        self.W_critic = s * rng.standard_normal((hidden, 1))
        self.b_critic = np.zeros(1)

    def forward(self, x):
        h = np.maximum(0, x @ self.W_shared + self.b_shared)
        logits = h @ self.W_actor + self.b_actor
        value = (h @ self.W_critic + self.b_critic).squeeze()
        return _softmax(logits), value

    def select_action(self, x, seed=0):
        probs, _ = self.forward(x)
        rng = np.random.default_rng(seed)
        return int(rng.choice(self.n_actions, p=probs / probs.sum())), probs

    def update(self, state, action, reward, next_state, done):
        """A2C update: TD(0) advantage, single step."""
        probs, value = self.forward(state)
        _, next_value = self.forward(next_state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value
        advantage = target - value
        # Actor: grad log pi * advantage
        grad_log = -probs.copy()
        grad_log[action] += 1.0
        # Critic: -2 * advantage * grad V
        # Numerico: solo ajustar W_critic
        h = np.maximum(0, state @ self.W_shared + self.b_shared)
        self.W_critic += self.lr * advantage * h.reshape(-1, 1)
        self.b_critic += self.lr * advantage
        # Actor
        self.W_actor += self.lr * advantage * np.outer(h, grad_log)
        self.b_actor += self.lr * advantage * grad_log
        return float(advantage)


def n_step_advantage(rewards, values, gamma, n, done_at):
    """N-step advantage: A_t = sum_{k=0}^{n-1} gamma^k r_{t+k+1} + gamma^n * V(s_{t+n}) - V(s_t).
    done_at: terminal step o -1 si continuing.
    """
    T = len(rewards)
    advantages = np.zeros(T)
    for t in range(T):
        G = 0
        for k in range(n):
            if t + k >= T:
                break
            G += gamma ** k * rewards[t + k]
        if t + n < T and (done_at < 0 or t + n <= done_at):
            G += gamma ** n * values[t + n]
        advantages[t] = G - values[t]
    return advantages


def gae_advantage(rewards, values, dones, gamma, lambda_):
    """GAE: A_t = sum_{k=0}^inf (gamma*lambda)^k * delta_{t+k}.
    delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t).
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lambda_ * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    return advantages


def a2c_a3c_summary():
    """Comparacion A2C vs A3C."""
    return {
        "A2C (Advantage Actor-Critic)": "Sync, single learner, GAE",
        "A3C (Asynchronous)": "Multiple async workers, each con su copy of env",
        "Distribucion": "A3C: workers update global en asynco. A2C: gradients sincronizados.",
        "Hoy": "A2C es standard, A3C en desuso (reemplazado por IMPALA, A2C con distribucion)",
        "Extension": "PPO, SAC, TD3 son evoluciones",
    }


def main() -> int:
    rng = np.random.default_rng(0)
    n_states, n_actions = 4, 2
    ac = ActorCritic(n_states, n_actions, hidden=8, seed=0)
    # Mock
    s = np.array([1, 0, 0, 0], dtype=float)
    a, probs = ac.select_action(s, seed=0)
    s_next = np.array([0, 1, 0, 0], dtype=float)
    r = 1.0
    adv = ac.update(s, a, r, s_next, done=False)
    print(f"Action: {a}, Advantage: {adv:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())