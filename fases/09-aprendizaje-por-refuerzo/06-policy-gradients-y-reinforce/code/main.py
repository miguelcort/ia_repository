"""
Lección: 06-policy-gradients-y-reinforce
Fase: 09
Policy gradient: V(s, theta) = sum_a pi(a|s, theta) * Q(s, a).
REINFORCE: grad J = E[grad log pi(a|s) * G_t].
"""
from __future__ import annotations
import sys
import numpy as np
import math


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def policy_forward(state, W, b, n_actions):
    """Softmax policy: pi(a|s) = softmax(state @ W + b)."""
    logits = state @ W + b
    return _softmax(logits)


def log_prob(action, probs):
    """log pi(a|s)."""
    return float(np.log(probs[action] + 1e-9))


def discounted_returns(rewards, gamma):
    """G_t = sum_{k=t}^T gamma^{k-t} r_k."""
    T = len(rewards)
    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = gamma * G + rewards[t]
        returns[t] = G
    return returns


def reinforce_episode(states, actions, rewards, W, b, gamma, lr):
    """REINFORCE update.
    states: (T, n_states). actions: (T,). rewards: (T,).
    W: (n_states, n_actions). b: (n_actions,).
    Returns: W, b updated.
    """
    returns = discounted_returns(rewards, gamma)
    T = len(states)
    for t in range(T):
        # Forward
        probs = policy_forward(states[t], W, b, n_actions=len(b))
        # grad log pi(a_t | s_t) = one_hot(a_t) - pi
        grad_log = -probs.copy()
        grad_log[actions[t]] += 1.0
        # Outer product para grad W
        grad_W = np.outer(states[t], grad_log)
        # Update con returns como baseline (no baseline en REINFORCE simple)
        W += lr * returns[t] * grad_W
        b += lr * returns[t] * grad_log
    return W, b


def baseline_returns(returns, baseline):
    """Returns - baseline (variance reduction)."""
    return returns - baseline


def policy_gradient_theorem():
    """Formula teorica."""
    return "grad J(theta) = E_pi[grad log pi(a|s) * Q(s, a)]"


def main() -> int:
    rng = np.random.default_rng(0)
    n_states, n_actions = 4, 2
    W = rng.standard_normal((n_states, n_actions)) * 0.1
    b = np.zeros(n_actions)
    gamma = 0.9
    lr = 0.01
    # Mock episode
    states = [np.array([1, 0, 0, 0]),
              np.array([0, 1, 0, 0]),
              np.array([0, 0, 1, 0]),
              np.array([0, 0, 0, 1])]
    actions = [0, 1, 0, 1]
    rewards = [0.0, 0.0, 0.0, 1.0]
    W, b = reinforce_episode(states, actions, rewards, W, b, gamma, lr)
    returns = discounted_returns(rewards, gamma)
    print(f"Returns: {returns.round(3)}")
    print(f"Final W norm: {np.linalg.norm(W):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())