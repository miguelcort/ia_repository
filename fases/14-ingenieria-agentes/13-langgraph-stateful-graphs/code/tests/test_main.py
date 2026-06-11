"""Pruebas para 13-langgraph-stateful-graphs."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestState(unittest.TestCase):
    def test_basic(self):
        s = main.State()
        s.set("k", "v")
        self.assertEqual(s.get("k"), "v")

    def test_get_default(self):
        s = main.State()
        self.assertEqual(s.get("missing", "default"), "default")

    def test_update(self):
        s = main.State()
        s.update({"a": 1, "b": 2})
        self.assertEqual(s.get("a"), 1)
        self.assertEqual(s.get("b"), 2)

    def test_history(self):
        s = main.State()
        s.set("k", "v")
        self.assertEqual(len(s.history), 1)

    def test_snapshot(self):
        s = main.State({"x": 1})
        s.set("y", 2)
        snap = s.snapshot()
        self.assertEqual(snap["x"], 1)
        self.assertEqual(snap["y"], 2)


class TestNode(unittest.TestCase):
    def test_run(self):
        n = main.Node("test", lambda s: s.set("done", True))
        s = main.State()
        n.run(s)
        self.assertTrue(s.get("done"))


class TestEdge(unittest.TestCase):
    def test_basic(self):
        e = main.Edge("a", "b")
        self.assertEqual(e.src, "a")
        self.assertEqual(e.dst, "b")

    def test_conditional(self):
        cond = lambda s: True
        e = main.Edge("a", "b", condition=cond)
        s = main.State()
        self.assertTrue(e.condition(s))


class TestGraph(unittest.TestCase):
    def test_add_node(self):
        g = main.Graph()
        g.add_node("a", lambda s: None)
        self.assertIn("a", g.nodes)

    def test_add_edge(self):
        g = main.Graph()
        g.add_edge("a", "b")
        self.assertEqual(len(g.edges["a"]), 1)

    def test_run_linear(self):
        g = main.Graph()
        g.set_entry("a")
        g.add_node("a", lambda s: s.set("step", 1))
        g.add_node("b", lambda s: s.set("step", 2))
        g.add_node("c", lambda s: s.set("step", 3))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "end_placeholder")  # end
        s = main.State()
        g.run(s, max_steps=10)
        self.assertEqual(s.get("step"), 3)

    def test_conditional(self):
        g = main.Graph()
        g.set_entry("a")
        g.add_node("a", lambda s: s.set("step", 1))
        g.add_node("b", lambda s: s.set("step", 2))
        g.add_node("c", lambda s: s.set("done", True))
        g.add_edge("a", "b")
        g.add_edge("b", "c", condition=lambda s: "step" in s.data)
        s = main.State()
        g.run(s, max_steps=10)
        self.assertTrue(s.get("done"))


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