"""
Lección: 12-rl-para-juegos
Fase: 09
RL para juegos: AlphaGo, AlphaZero, MuZero, AlphaStar, OpenAI Five.
Self-play, MCTS + NN, learning from scratch.
"""
from __future__ import annotations
import sys
import numpy as np


def mcts_select(node, c_puct=1.0):
    """Select child con UCB (PUCT formula).
    Q + c_puct * P * sqrt(N_parent) / (1 + N_child).
    """
    best_score = -np.inf
    best_child = None
    for action, child in node.children.items():
        u = child.q_value + c_puct * child.prior * np.sqrt(node.n_visits) / (1 + child.n_visits)
        if u > best_score:
            best_score = u
            best_child = child
    return best_child


def mcts_backprop(path, value):
    """Backprop value a lo largo del path."""
    for node in reversed(path):
        node.n_visits += 1
        node.value_sum += value
        node.q_value = node.value_sum / node.n_visits
        value = -value  # alternar para 2-player


def self_play_iteration(policy_eval, opponent_eval, n_games, win_rate_threshold=0.5):
    """Self-play iteration. Si policy gana > threshold vs opponent, es nueva best.
    """
    wins = 0
    for g in range(n_games):
        if policy_eval() > opponent_eval():
            wins += 1
    win_rate = wins / n_games
    return win_rate >= win_rate_threshold


def alphazero_components():
    """AlphaZero components."""
    return {
        "MCTS": "Monte Carlo Tree Search, PUCT selection",
        "Policy + Value net": "Input state, output (pi, v)",
        "Self-play": "Generate games via MCTS, train on them",
        "Replay buffer": "Store games, sample minibatches",
        "Muzero": "Learned model, MCTS en latent",
        "AlphaStar": "StarCraft, supervised + RL",
    }


def game_complexity():
    """Complejidad de juegos."""
    return {
        "Tic-tac-toe": "10^3 states, resuelto",
        "Connect Four": "10^13, AlphaBeta search",
        "Chess": "10^44, AlphaZero dominates",
        "Go": "10^170, AlphaGo -> AlphaZero -> KataGo",
        "StarCraft 2": "10^1685 actions, AlphaStar (2019)",
        "Dota 2": "Long horizon, OpenAI Five (2019)",
    }


def main() -> int:
    print("=== Game complexity ===")
    for k, v in game_complexity().items():
        print(f"  {k:18s} {v}")
    print("\n=== AlphaZero components ===")
    for k, v in alphazero_components().items():
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())