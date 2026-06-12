"""Pruebas para 24-chaos-engineering-llm."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestExperiment(unittest.TestCase):
    def test_create(self):
        e = main.ChaosExperiment("x", "hypothesis")
        self.assertEqual(e.name, "x")

    def test_run_no_injectors(self):
        e = main.ChaosExperiment("x", "h")
        rate = e.run(lambda: 1, n_trials=5)
        self.assertEqual(rate, 1.0)

    def test_run_with_error(self):
        e = main.ChaosExperiment("x", "h")
        e.add_injector(main.inject_error)
        rate = e.run(lambda: 1, n_trials=5)
        self.assertEqual(rate, 0.0)


class TestInjectors(unittest.TestCase):
    def test_inject_error(self):
        with self.assertRaises(RuntimeError):
            main.inject_error()

    def test_inject_malformed(self):
        r = main.inject_malformed_response()
        self.assertIn("unexpected_field", r)


class TestMonkey(unittest.TestCase):
    def test_register_run(self):
        m = main.ChaosMonkey(seed=42)
        e = main.ChaosExperiment("x", "h")
        m.register(e)
        results = m.run_all(lambda: 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["success_rate"], 1.0)

    def test_multiple(self):
        m = main.ChaosMonkey(seed=42)
        m.register(main.ChaosExperiment("a", "h"))
        m.register(main.ChaosExperiment("b", "h"))
        results = m.run_all(lambda: 1)
        self.assertEqual(len(results), 2)


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