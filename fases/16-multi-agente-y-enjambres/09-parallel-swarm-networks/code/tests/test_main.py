"""Pruebas para 09-parallel-swarm-networks."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFanOut(unittest.TestCase):
    def test_basic(self):
        agents = [lambda t: f"a{i}:{t}" for i in range(3)]
        results = main.fan_out(agents, "x")
        self.assertEqual(len(results), 3)

    def test_empty(self):
        results = main.fan_out([], "x")
        self.assertEqual(results, [])


class TestFanIn(unittest.TestCase):
    def test_concat(self):
        results = [{"result": "a"}, {"result": "b"}]
        self.assertEqual(main.fan_in(results, "concat"), ["a", "b"])

    def test_first(self):
        results = [{"result": "a"}, {"result": "b"}]
        self.assertEqual(main.fan_in(results, "first"), "a")

    def test_first_empty(self):
        self.assertIsNone(main.fan_in([], "first"))

    def test_majority(self):
        results = [{"result": "a"}, {"result": "a"}, {"result": "b"}]
        self.assertEqual(main.fan_in(results, "majority"), "a")

    def test_majority_empty(self):
        self.assertIsNone(main.fan_in([], "majority"))


class TestSwarm(unittest.TestCase):
    def test_create(self):
        s = main.Swarm([lambda t: t])
        self.assertEqual(len(s.agents), 1)

    def test_run_concat(self):
        agents = [lambda t: f"r{i}:{t}" for i in range(3)]
        s = main.Swarm(agents, reducer="concat")
        result = s.run("x")
        self.assertEqual(len(result), 3)

    def test_run_first(self):
        s = main.Swarm([lambda t: "a", lambda t: "b"], reducer="first")
        self.assertEqual(s.run("x"), "a")

    def test_run_many(self):
        s = main.Swarm([lambda t: t], reducer="first")
        results = s.run_many(["a", "b", "c"])
        self.assertEqual(results, ["a", "b", "c"])

    def test_history(self):
        s = main.Swarm([lambda t: t])
        s.run("x")
        self.assertEqual(len(s.history), 1)


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