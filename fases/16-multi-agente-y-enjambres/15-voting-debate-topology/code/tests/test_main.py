"""Pruebas para 15-voting-debate-topology."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestWeightedVote(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(main.weighted_vote(["a", "b", "a"], [1, 2, 3]), "a")

    def test_empty(self):
        self.assertIsNone(main.weighted_vote([], []))


class TestApproveVote(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(main.approve_vote(["yes", "yes", "no"], threshold=0.5))

    def test_fail(self):
        self.assertFalse(main.approve_vote(["no", "no", "yes"], threshold=0.5))

    def test_empty(self):
        self.assertFalse(main.approve_vote([]))


class TestTopologies(unittest.TestCase):
    def test_ring(self):
        edges = main.ring_topology(4)
        self.assertEqual(len(edges), 4)
        self.assertIn((0, 1), edges)
        self.assertIn((3, 0), edges)

    def test_star(self):
        edges = main.star_topology(4)
        self.assertEqual(len(edges), 3)
        for e in edges:
            self.assertEqual(e[0], 0)

    def test_fully_connected(self):
        edges = main.fully_connected_topology(4)
        self.assertEqual(len(edges), 6)


class TestDebateTopology(unittest.TestCase):
    def test_basic(self):
        history = main.debate_topology(["a", "b", "a"], rounds=2)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], ["a", "b", "a"])


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