"""Pruebas para 07-tensorrt-llm-blackwell."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestGPUs(unittest.TestCase):
    def test_list(self):
        gpus = main.list_gpus()
        self.assertIn("A100", gpus)
        self.assertIn("H100", gpus)
        self.assertIn("B100", gpus)
        self.assertIn("B200", gpus)

    def test_get(self):
        g = main.get_gpu("B200")
        self.assertEqual(g["architecture"], "Blackwell")
        self.assertEqual(g["memory_gb"], 192)

    def test_get_unknown(self):
        self.assertIsNone(main.get_gpu("unknown"))


class TestBest(unittest.TestCase):
    def test_throughput(self):
        name, info = main.best_for_throughput()
        self.assertEqual(name, "B200")

    def test_memory(self):
        name, info = main.best_for_memory()
        self.assertEqual(name, "B100")


class TestCanRun(unittest.TestCase):
    def test_can_run_small(self):
        self.assertTrue(main.can_run_model("B200", 70))

    def test_cannot_run_large(self):
        self.assertFalse(main.can_run_model("A100", 70))

    def test_unknown(self):
        self.assertFalse(main.can_run_model("unknown", 7))


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