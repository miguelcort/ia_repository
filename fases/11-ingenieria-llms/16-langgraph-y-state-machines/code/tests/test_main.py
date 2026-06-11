"""Pruebas para 16-langgraph-y-state-machines."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestNode(unittest.TestCase):
    def test_basic(self):
        s = main.graph_node({"messages": ["a"]})
        self.assertEqual(s["messages"], ["a", "processed"])


class TestEdge(unittest.TestCase):
    def test_unconditional(self):
        self.assertTrue(main.graph_edge("a", "b"))

    def test_conditional(self):
        self.assertTrue(main.graph_edge("a", "b", lambda: True))
        self.assertFalse(main.graph_edge("a", "b", lambda: False))


class TestStateInit(unittest.TestCase):
    def test_keys(self):
        s = main.graph_state_init()
        self.assertIn("messages", s)
        self.assertEqual(s["step"], 0)


class TestBuildGraph(unittest.TestCase):
    def test_basic(self):
        nodes, edges = main.build_simple_graph()
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(edges), 2)


class TestGraphStep(unittest.TestCase):
    def test_basic(self):
        state = {"step": 0}
        def transition(s):
            s = {**s, "step": s["step"] + 1}
            if s["step"] >= 5:
                s["done"] = True
            return s
        state = main.graph_step(state, transition, max_steps=10)
        self.assertEqual(state["step"], 5)
        self.assertTrue(state["done"])


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