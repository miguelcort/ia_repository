"""Pruebas para 17-generative-agents-simulation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestGenerativeAgent(unittest.TestCase):
    def test_create(self):
        a = main.GenerativeAgent("alice", "friendly")
        self.assertEqual(a.name, "alice")
        self.assertEqual(a.memory_stream, [])

    def test_observe(self):
        a = main.GenerativeAgent("alice", "x")
        e = a.observe("walked in park")
        self.assertEqual(e["event"], "walked in park")
        self.assertIn("importance", e)
        self.assertEqual(len(a.memory_stream), 1)

    def test_importance_scoring(self):
        a = main.GenerativeAgent("alice", "x")
        e1 = a.observe("walked")
        e2 = a.observe("met love of my life")
        self.assertGreater(e2["importance"], e1["importance"])

    def test_reflect(self):
        a = main.GenerativeAgent("alice", "x")
        a.observe("e1")
        a.observe("e2")
        r = a.reflect()
        self.assertIsNotNone(r)
        self.assertEqual(len(a.reflections), 1)

    def test_reflect_empty(self):
        a = main.GenerativeAgent("alice", "x")
        self.assertIsNone(a.reflect())

    def test_plan(self):
        a = main.GenerativeAgent("alice", "x")
        plan = a.plan_day(["greet", "work", "rest"])
        self.assertEqual(len(plan), 3)

    def test_recall(self):
        a = main.GenerativeAgent("alice", "x")
        a.observe("walked dog")
        a.observe("ate lunch")
        results = a.recall("dog")
        self.assertEqual(len(results), 1)


class TestSimulate(unittest.TestCase):
    def test_simulate(self):
        a1 = main.GenerativeAgent("alice", "x")
        a2 = main.GenerativeAgent("bob", "y")
        events = main.simulate_day([a1, a2], hours=12)
        self.assertEqual(len(events), 24)
        self.assertGreater(len(a1.reflections), 0)


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