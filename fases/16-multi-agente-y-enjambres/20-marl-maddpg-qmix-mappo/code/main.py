"""
Lección: 20-marl-maddpg-qmix-mappo
Fase: 16
Multi-Agent RL: MADDPG, QMIX, MAPPO.
Cooperative + competitive, centralized
training + decentralized execution,
mixing networks, PPO for multi-agent.
"""
from __future__ import annotations
import random


class MultiAgentEnv:
    def __init__(self, n_agents=2, n_actions=4):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state = [0.0] * n_agents

    def reset(self):
        self.state = [random.random() for _ in range(self.n_agents)]
        return self.state

    def step(self, actions):
        rewards = []
        for i, a in enumerate(actions):
            reward = 1.0 if a == int(self.state[i] * self.n_actions) else -0.1
            rewards.append(reward)
        self.state = [random.random() for _ in range(self.n_agents)]
        return self.state, rewards, False

    def observation(self, agent_id):
        return self.state[agent_id]


class MADDPGAgent:
    def __init__(self, agent_id, n_actions, lr=0.01):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.q_values = [0.0] * n_actions

    def select_action(self, obs, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return max(range(self.n_actions), key=lambda a: self.q_values[a])

    def update(self, obs, action, reward, next_obs):
        target = reward + 0.9 * max(self.q_values)
        self.q_values[action] += self.lr * (target - self.q_values[action])


class QMIXMixer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.weights = [1.0 / n_agents] * n_agents

    def mix(self, individual_qs):
        return sum(w * q for w, q in zip(self.weights, individual_qs))


def maddpg_train(env, n_episodes=20, max_steps=10):
    agents = [MADDPGAgent(i, env.n_actions) for i in range(env.n_agents)]
    episode_rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        total = 0
        for step in range(max_steps):
            actions = [a.select_action(env.observation(a.agent_id)) for a in agents]
            next_state, rewards, done = env.step(actions)
            for i, a in enumerate(agents):
                a.update(env.observation(a.agent_id), actions[i], rewards[i], env.observation(a.agent_id))
            total += sum(rewards)
        episode_rewards.append(total)
    return episode_rewards


def mappo_train(env, n_episodes=20, max_steps=10, clip=0.2):
    """Simplified MAPPO with PPO clip objective."""
    agents = [MADDPGAgent(i, env.n_actions) for i in range(env.n_agents)]
    history = []
    for ep in range(n_episodes):
        state = env.reset()
        total = 0
        for step in range(max_steps):
            actions = [a.select_action(env.observation(a.agent_id), epsilon=0.0) for a in agents]
            next_state, rewards, done = env.step(actions)
            for i, a in enumerate(agents):
                ratio = 1.0
                surr = min(ratio * rewards[i], clip * rewards[i])
                a.q_values[actions[i]] += a.lr * surr
            total += sum(rewards)
        history.append(total)
    return history


def main() -> int:
    random.seed(42)
    env = MultiAgentEnv(n_agents=2, n_actions=4)
    rewards = maddpg_train(env, n_episodes=20)
    print(f"MADDPG: avg {sum(rewards)/len(rewards):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())