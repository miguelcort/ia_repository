"""Pruebas para 03-blip2-y-qformer-bridge."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestQFormerQueries(unittest.TestCase):
    def test_shapes(self):
        q, Wq, Wkv = main.qformer_queries(num_queries=32, query_dim=768, seed=0)
        self.assertEqual(q.shape, (32, 768))
        self.assertEqual(Wq.shape, (768, 768))
        self.assertEqual(Wkv.shape, (768, 768))


class TestCrossAttention(unittest.TestCase):
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        queries = rng.standard_normal((32, 768)) * 0.02
        kv = rng.standard_normal((257, 768)) * 0.1
        Wq = rng.standard_normal((768, 768)) * 0.1
        Wkv = rng.standard_normal((768, 768)) * 0.1
        out = main.cross_attention(queries, kv, Wq, Wkv)
        self.assertEqual(out.shape, (32, 768))

    def test_output_shape_mismatched_dim(self):
        rng = np.random.default_rng(0)
        queries = rng.standard_normal((32, 768)) * 0.02
        kv = rng.standard_normal((100, 512)) * 0.1
        Wq = rng.standard_normal((768, 768)) * 0.1
        Wkv = rng.standard_normal((768, 768)) * 0.1
        out = main.cross_attention(queries, kv, Wq, Wkv)
        self.assertEqual(out.shape, (32, 768))


class TestQFormerForward(unittest.TestCase):
    def test_layers_compose(self):
        rng = np.random.default_rng(0)
        image = rng.standard_normal((257, 768)) * 0.1
        queries, Wq, Wkv = main.qformer_queries(num_queries=32, query_dim=768, seed=0)
        z1 = main.qformer_forward(image, queries, Wq, Wkv, n_layers=1)
        z3 = main.qformer_forward(image, queries, Wq, Wkv, n_layers=3)
        # 3 layers must not equal 1 layer
        self.assertFalse(np.allclose(z1, z3))


class TestBLIP2Forward(unittest.TestCase):
    def test_concat_image_text(self):
        rng = np.random.default_rng(0)
        image = rng.standard_normal((257, 768)) * 0.1
        text = rng.standard_normal((10, 4096)) * 0.1
        queries, Wq, Wkv = main.qformer_queries(num_queries=32, query_dim=768, seed=0)
        llm_proj = rng.standard_normal((768, 4096)) * 0.02
        out = main.blip2_forward(image, text, queries, Wq, Wkv, llm_proj)
        # 32 image tokens + 10 text = 42
        self.assertEqual(out.shape, (42, 4096))


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