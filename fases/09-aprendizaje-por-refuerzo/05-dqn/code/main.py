"""
Lección: 05-dqn
Fase: 09
DQN: Deep Q-Network. Neural network para Q function, experience replay,
target network. Atari DQN.
"""
from __future__ import annotations
import sys
import numpy as np
import math


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(0, x)


class DQN:
    """Deep Q-Network mock. Replay buffer, target network, epsilon decay."""

    def __init__(self, n_states, n_actions, hidden=64, seed=0, gamma=0.99, lr=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_states)
        # Q-network params
        self.W1 = s * rng.standard_normal((n_states, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = s * rng.standard_normal((hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        # Target network (copy)
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        # Replay buffer
        self.replay = []
        self.epsilon = 1.0

    def forward(self, x, params=None):
        """x: (n_states,). Returns Q-values (n_actions,)."""
        if params is None:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        else:
            W1, b1, W2, b2 = params
        h = _relu(x @ W1 + b1)
        return h @ W2 + b2

    def target_forward(self, x):
        return self.forward(x, (self.W1_t, self.b1_t, self.W2_t, self.b2_t))

    def act(self, x, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        rng = np.random.default_rng()
        if rng.uniform() < epsilon:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(self.forward(x)))

    def store(self, transition):
        """transition: (s, a, r, s_next, done)."""
        self.replay.append(transition)
        # Capacidad: mantener ultimas 10K
        if len(self.replay) > 10000:
            self.replay.pop(0)

    def sample_batch(self, batch_size=32, seed=0):
        rng = np.random.default_rng(seed)
        if len(self.replay) < batch_size:
            return None
        idx = rng.choice(len(self.replay), size=batch_size, replace=False)
        return [self.replay[i] for i in idx]

    def train_step(self, batch_size=32, seed=0):
        batch = self.sample_batch(batch_size, seed=seed)
        if batch is None:
            return 0.0
        loss = 0.0
        for (s, a, r, s_next, done) in batch:
            target = r + (0 if done else self.gamma * np.max(self.target_forward(s_next)))
            q_pred = self.forward(np.asarray(s, dtype=float))[a]
            loss += (target - q_pred) ** 2
        loss /= batch_size
        # Gradient update (numerico, simple)
        # En practice: backprop
        for (s, a, r, s_next, done) in batch:
            target = r + (0 if done else self.gamma * np.max(self.target_forward(s_next)))
            x = np.asarray(s, dtype=float)
            # d/dW2 (output layer)
            h = _relu(x @ self.W1 + self.b1)
            q = h @ self.W2 + self.b2
            dL_dq = 2 * (q[a] - target) / batch_size
            dL_dW2 = np.outer(h, np.zeros_like(self.W2[:, a]))
            dL_dW2[:, a] = h * dL_dq
            dL_db2 = np.zeros_like(self.b2)
            dL_db2[a] = dL_dq
            # Skip hidden backprop para demo
            self.W2 -= self.lr * dL_dW2
            self.b2 -= self.lr * dL_db2
        return float(loss)

    def update_target(self):
        """Copy Q-net weights to target net."""
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()

    def decay_epsilon(self, decay=0.995, min_eps=0.01):
        self.epsilon = max(min_eps, self.epsilon * decay)


def prioritized_experience_weights(td_errors, alpha=0.6):
    """Prioritized experience replay: p_i = |td_i|^alpha + eps.
    Returns: weights (probabilities).
    """
    p = np.abs(td_errors) ** alpha + 1e-6
    return p / p.sum()


def dqn_components():
    """DQN components SOTA."""
    return {
        "Experience replay": "Buffer de (s, a, r, s', done), samplea batches",
        "Target network": "Q-target con weights frozen, updated every N steps",
        "Epsilon decay": "epsilon 1.0 -> 0.1 over training",
        "Loss": "MSE entre Q(s, a) y r + gamma * max Q_target(s', a')",
        "Frame stacking": "Stack ultimas 4 frames como input",
        "Reward clipping": "Clip reward a [-1, 1]",
    }


def main() -> int:
    n_states, n_actions = 4, 2
    agent = DQN(n_states, n_actions, hidden=8, seed=0)
    # Demo
    for ep in range(5):
        s = np.zeros(n_states)
        s[0] = 1.0
        a = agent.act(s)
        r = 1.0
        s_next = np.zeros(n_states)
        s_next[1] = 1.0
        agent.store((s, a, r, s_next, False))
    print(f"Replay size: {len(agent.replay)}")
    # Demo: solo print stats
    print(f"Replay size: {len(agent.replay)}, Epsilon: {agent.epsilon:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())