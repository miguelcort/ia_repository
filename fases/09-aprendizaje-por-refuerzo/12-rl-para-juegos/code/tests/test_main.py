"""Pruebas para 12-rl-para-juegos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMCTSSelect(unittest.TestCase):
    def test_select(self):
        class Node:
            def __init__(self, q, n, p):
                self.q_value = q
                self.n_visits = n
                self.prior = p
                self.children = {}
        class Tree:
            def __init__(self):
                self.n_visits = 10
                self.children = {
                    0: Node(0.5, 5, 0.3),
                    1: Node(0.3, 3, 0.5),
                }
        tree = Tree()
        # Just verify the function returns one of the children
        best = main.mcts_select(tree, c_puct=1.0)
        self.assertIn(best, [tree.children[0], tree.children[1]])


class TestMCTSBackprop(unittest.TestCase):
    def test_backprop_updates(self):
        class Node:
            def __init__(self):
                self.n_visits = 0
                self.value_sum = 0
                self.q_value = 0
        path = [Node() for _ in range(3)]
        value = 1.0
        main.mcts_backprop(path, value)
        # First node: +1 visit, q=1
        self.assertEqual(path[0].n_visits, 1)
        # Last node: value=1, then alternated
        self.assertEqual(path[2].n_visits, 1)
        # q_value debe ser 1
        self.assertEqual(path[0].q_value, 1.0)


class TestSelfPlay(unittest.TestCase):
    def test_threshold(self):
        # Si policy gana > 50% sobre 10 games, devuelve True
        def policy():
            return 0.6
        def opponent():
            return 0.4
        result = main.self_play_iteration(policy, opponent, 10, win_rate_threshold=0.5)
        self.assertTrue(result)

    def test_below_threshold(self):
        def policy():
            return 0.3
        def opponent():
            return 0.7
        result = main.self_play_iteration(policy, opponent, 10, win_rate_threshold=0.5)
        self.assertFalse(result)


class TestComponents(unittest.TestCase):
    def test_alphazero_seis(self):
        c = main.alphazero_components()
        self.assertEqual(len(c), 6)

    def test_games_seis(self):
        c = main.game_complexity()
        self.assertEqual(len(c), 6)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()