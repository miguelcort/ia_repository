"""
Lección: 02-programacion-dinamica
Fase: 09
Value iteration y policy iteration para resolver MDPs exactos.
"""
from __future__ import annotations
import sys
import numpy as np


def value_iteration(P, R, gamma, n_states, n_actions, theta=1e-6, max_iter=1000):
    """Value iteration: V_{k+1} = max_a sum_s' P(s'|s,a) * (R + gamma * V_k(s'))
    Returns: V, policy.
    """
    V = np.zeros(n_states)
    for it in range(max_iter):
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s2 in range(n_states):
                    q_values[a] += P[s, a, s2] * (R[s, a] + gamma * V[s2])
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s2 in range(n_states):
                q_values[a] += P[s, a, s2] * (R[s, a] + gamma * V[s2])
        policy[s] = int(np.argmax(q_values))
    return V, policy


def policy_iteration(P, R, gamma, n_states, n_actions, theta=1e-6, max_iter=100):
    """Policy iteration: eval + improve. Converge en pocas iteraciones."""
    policy = np.zeros(n_states, dtype=int)
    for it in range(max_iter):
        # Policy evaluation
        V = np.zeros(n_states)
        for _ in range(1000):
            delta = 0
            for s in range(n_states):
                v = V[s]
                a = policy[s]
                V[s] = sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(n_states))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        # Policy improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s2 in range(n_states):
                    q_values[a] += P[s, a, s2] * (R[s, a] + gamma * V[s2])
            policy[s] = int(np.argmax(q_values))
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return V, policy


def gridworld_value(n=4):
    """4x4 gridworld, reward +1 en (0,3), -1 en (0,0), -0.04 elsewhere.
    Acciones: up/down/left/right.
    Returns: P, R, goal, pit.
    """
    n_states = n * n
    n_actions = 4  # 0=up, 1=right, 2=down, 3=left
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    goal = 3
    pit = 0
    for r in range(n):
        for c in range(n):
            s = r * n + c
            for a in range(n_actions):
                # 0=up, 1=right, 2=down, 3=left
                nr, nc = r, c
                if a == 0:
                    nr = max(0, r - 1)
                elif a == 1:
                    nc = min(n - 1, c + 1)
                elif a == 2:
                    nr = min(n - 1, r + 1)
                elif a == 3:
                    nc = max(0, c - 1)
                s2 = nr * n + nc
                P[s, a, s2] = 1.0
                if s2 == goal:
                    R[s, a] = 1.0
                elif s2 == pit:
                    R[s, a] = -1.0
                else:
                    R[s, a] = -0.04
    return P, R, n_states, n_actions


def main() -> int:
    P, R, n_s, n_a = gridworld_value(4)
    V, policy = value_iteration(P, R, gamma=0.9, n_states=n_s, n_actions=n_a)
    print(f"Gridworld 4x4 value function (reshape a 4x4):")
    print(V.reshape(4, 4).round(2))
    print(f"\nPolicy (reshape a 4x4, 0=up, 1=right, 2=down, 3=left):")
    print(policy.reshape(4, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())