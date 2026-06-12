"""Pruebas para 10-cold-start-mitigation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestModelPool(unittest.TestCase):
    def test_warm(self):
        p = main.ModelPool(max_size=3)
        self.assertTrue(p.warm("m1", lambda: "M1"))
        self.assertEqual(p.size(), 1)

    def test_warm_idempotent(self):
        p = main.ModelPool(max_size=3)
        p.warm("m1", lambda: "M1")
        self.assertFalse(p.warm("m1", lambda: "M1"))
        self.assertEqual(p.size(), 1)

    def test_get(self):
        p = main.ModelPool(max_size=3)
        p.warm("m1", lambda: "M1")
        self.assertEqual(p.get("m1"), "M1")

    def test_get_missing(self):
        p = main.ModelPool(max_size=3)
        self.assertIsNone(p.get("missing"))

    def test_evict_idle(self):
        p = main.ModelPool(max_size=1, idle_timeout=0.01)
        p.warm("m1", lambda: "M1")
        time.sleep(0.1)
        p.warm("m2", lambda: "M2")
        self.assertEqual(p.size(), 1)
        self.assertIn("m2", p.loaded)


import time


class TestPredictive(unittest.TestCase):
    def test_record_predict(self):
        p = main.ModelPool()
        w = main.PredictiveWarmer(p)
        w.record_request("m1")
        w.record_request("m1")
        w.record_request("m2")
        predicted = w.predict_next_models(top_k=1)
        self.assertEqual(predicted[0], "m1")

    def test_empty(self):
        p = main.ModelPool()
        w = main.PredictiveWarmer(p)
        self.assertEqual(w.predict_next_models(), [])

    def test_warm_predicted(self):
        p = main.ModelPool()
        w = main.PredictiveWarmer(p)
        w.record_request("m1")
        w.warm_predicted(lambda name: f"loaded-{name}")
        self.assertIn("m1", p.loaded)


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