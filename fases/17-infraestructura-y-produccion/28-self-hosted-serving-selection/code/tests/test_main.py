"""Pruebas para 28-self-hosted-serving-selection."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestOptions(unittest.TestCase):
    def test_list(self):
        opts = main.list_options()
        self.assertIn("vllm", opts)
        self.assertIn("tgi", opts)
        self.assertIn("llama_cpp", opts)

    def test_get(self):
        o = main.get_option("vllm")
        self.assertEqual(o["name"], "vLLM")
        self.assertTrue(o["gpu_required"])

    def test_get_unknown(self):
        self.assertIsNone(main.get_option("unknown"))


class TestByThroughput(unittest.TestCase):
    def test_high(self):
        result = main.by_throughput("high")
        self.assertIn("vllm", result)

    def test_very_high(self):
        result = main.by_throughput("very_high")
        self.assertIn("tensorrt_llm", result)


class TestRecommend(unittest.TestCase):
    def test_no_gpu(self):
        rec = main.recommend(have_gpu=False)
        self.assertEqual(rec, "llama_cpp")

    def test_gpu_high_ease(self):
        rec = main.recommend(have_gpu=True, ease="high")
        self.assertIsNotNone(rec)


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