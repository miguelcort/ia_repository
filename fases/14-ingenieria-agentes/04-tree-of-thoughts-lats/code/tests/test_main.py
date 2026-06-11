"""Pruebas para 04-tree-of-thoughts-lats."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestTreeNode(unittest.TestCase):
    def test_basic(self):
        n = main.TreeNode("state")
        self.assertEqual(n.state, "state")
        self.assertTrue(n.is_leaf())

    def test_add_child(self):
        n = main.TreeNode("a")
        c = main.TreeNode("b", parent=n)
        n.add_child(c)
        self.assertFalse(n.is_leaf())
        self.assertIn(c, n.children)

    def test_path(self):
        a = main.TreeNode("a")
        b = main.TreeNode("b", parent=a)
        c = main.TreeNode("c", parent=b)
        p = c.path()
        self.assertEqual([n.state for n in p], ["a", "b", "c"])


class TestBFS(unittest.TestCase):
    def test_basic(self):
        root = main.TreeNode("start")
        expand = lambda s: [{"next_state": s + "1"}, {"next_state": s + "2"}]
        def evaluate(s):
            if "1" in s:
                return "goal"
            return "continue"
        result = main.bfs_expand(root, expand, evaluate, max_depth=3)
        self.assertIsNotNone(result)


class TestUCB(unittest.TestCase):
    def test_unvisited(self):
        n = main.TreeNode("a", value=1.0)
        n.visits = 0
        n.parent = main.TreeNode("root")
        n.parent.visits = 10
        self.assertEqual(main.ucb_select(n), float("inf"))

    def test_visited(self):
        n = main.TreeNode("a", value=10.0)
        n.visits = 5
        n.parent = main.TreeNode("root")
        n.parent.visits = 10
        score = main.ucb_select(n)
        self.assertLess(score, float("inf"))


class TestMCTS(unittest.TestCase):
    def test_basic(self):
        root = main.TreeNode("start")
        expand = lambda s: [{"next_state": s + "_good"}, {"next_state": s + "_bad"}]
        evaluate = lambda s: 1.0 if "good" in s else 0.0
        best = main.mcts(root, expand, evaluate, evaluate, n_iterations=20)
        # MCTS should prefer "good" branch
        self.assertIn("good", best.state)


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