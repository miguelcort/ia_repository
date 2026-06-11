"""Pruebas para 28-orchestration-patterns."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = main.Orchestrator()
        self.orch.register_worker("w1", lambda x: f"r1_{x}")
        self.orch.register_worker("w2", lambda x: f"r2_{x}")

    def test_register(self):
        self.assertIn("w1", self.orch.workers)
        self.assertIn("w2", self.orch.workers)

    def test_manager_worker(self):
        results = self.orch.manager_worker("task", n_workers=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r.startswith("r1_") or r.startswith("r2_"))

    def test_pipeline(self):
        result = self.orch.pipeline(["w1", "w2"], "data")
        # w1 first, then w2 -> r2 wraps r1
        self.assertTrue(result.startswith("r2_r1_"))

    def test_scatter_gather(self):
        result = self.orch.scatter_gather("task", n_workers=3)
        self.assertEqual(result["task"], "task")
        self.assertEqual(len(result["results"]), 3)


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