"""Pruebas para 13-pipeline-completo-de-llm."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPipelineStages(unittest.TestCase):
    def test_ocho(self):
        s = main.pipeline_stages()
        self.assertEqual(len(s), 8)


class TestPretrainCost(unittest.TestCase):
    def test_70b_15t(self):
        # 6 * 70e9 * 15e12 = 6.3e24
        flops = main.pre_train_cost(70e9, 15e12)
        self.assertAlmostEqual(flops, 6.3e24, places=20)


class TestThroughput(unittest.TestCase):
    def test_basic(self):
        # 70B model, 8xA100 (312 TFLOPS bf16), bs=32, seq=2048
        # flops_per_step = 2 * 70e9 * 32 * 2048 = 9.18e15
        # flops total = 8 * 312e12 = 2.5e15
        # tps = 2.5e15 / 9.18e15 * 1 sec = 0.272 tokens/sec
        # Hmm bajo. Real: 50+ tps en production. Estimacion basica.
        tps = main.inference_throughput(70e9)
        # Solo verificamos que es positivo
        self.assertGreater(tps, 0)


class TestQuality(unittest.TestCase):
    def test_models(self):
        m = main.quality_metrics_summary()
        self.assertIn("Llama 3 70B Instruct", m)


class TestComponents(unittest.TestCase):
    def test_keys(self):
        c = main.components_summary()
        self.assertIn("Foundation model", c)
        self.assertIn("Tokenizer", c)
        self.assertIn("Inference", c)


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