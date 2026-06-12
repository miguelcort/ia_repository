"""Pruebas para 21-metr-external-evaluation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBenchmarks(unittest.TestCase):
    def test_list(self):
        benchmarks = main.list_benchmarks()
        self.assertIn("HCAST", benchmarks)
        self.assertIn("RE-Bench", benchmarks)
        self.assertIn("SWE-bench", benchmarks)
        self.assertIn("GAIA", benchmarks)

    def test_get(self):
        b = main.get_benchmark("HCAST")
        self.assertIn("metric", b)
        self.assertEqual(b["metric"], "time_horizon")

    def test_get_unknown(self):
        self.assertIsNone(main.get_benchmark("unknown"))


class TestTimeHorizon(unittest.TestCase):
    def test_basic(self):
        tasks = {"t1": 60, "t2": 300, "t3": 600}
        model_times = {"t1": 10, "t2": 60, "t3": 120}
        horizon = main.time_horizon_score(tasks, model_times)
        self.assertGreater(horizon, 1)

    def test_empty(self):
        self.assertEqual(main.time_horizon_score({}, {}), 0.0)

    def test_partial(self):
        tasks = {"t1": 60, "t2": 300}
        model_times = {"t1": 10}
        horizon = main.time_horizon_score(tasks, model_times)
        self.assertEqual(horizon, 6.0)

    def test_median_odd(self):
        tasks = {"t1": 10, "t2": 20, "t3": 30}
        model_times = {"t1": 1, "t2": 2, "t3": 3}
        self.assertEqual(main.time_horizon_score(tasks, model_times), 10.0)

    def test_median_even(self):
        tasks = {"t1": 10, "t2": 20}
        model_times = {"t1": 1, "t2": 2}
        self.assertEqual(main.time_horizon_score(tasks, model_times), 10.0)


class TestEvaluateModel(unittest.TestCase):
    def test_hcast(self):
        result = main.evaluate_model({"HCAST": 0.75}, "HCAST")
        self.assertEqual(result["metric"], "time_horizon")
        self.assertEqual(result["value"], 0.75)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            main.evaluate_model({}, "unknown")

    def test_missing(self):
        result = main.evaluate_model({}, "HCAST")
        self.assertEqual(result["value"], 0.0)


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