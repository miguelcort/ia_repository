"""Pruebas para 16-pipeline-de-vision-capstone."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestInference(unittest.TestCase):
    def test_batch(self):
        imgs = [np.random.default_rng(i).normal(size=(10, 10, 3)) for i in range(8)]
        logits = main.stage_inference(imgs, batch_size=4)
        self.assertEqual(logits.shape, (8, 1000))

    def test_batch_unico(self):
        imgs = [np.random.default_rng(0).normal(size=(10, 10, 3))]
        logits = main.stage_inference(imgs, batch_size=1)
        self.assertEqual(logits.shape, (1, 1000))


class TestPostProcess(unittest.TestCase):
    def test_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        out = main.stage_postprocess(logits, top_k=2)
        # clase 1 (score mas alto) primero
        self.assertEqual(out[0][0][0], 1)
        self.assertGreater(out[0][0][1], out[0][1][1])


class TestLatency(unittest.TestCase):
    def test_p99_mayor_p50(self):
        latencias = [10, 12, 11, 15, 9, 14, 13, 50, 11, 10]
        stats = main.latency_p50_p99(latencias)
        self.assertGreater(stats["p99_ms"], stats["p50_ms"])


class TestThroughput(unittest.TestCase):
    def test_throughput(self):
        t = main.throughput(50, batch_size=4)
        self.assertEqual(t, 200)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Throughput", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()