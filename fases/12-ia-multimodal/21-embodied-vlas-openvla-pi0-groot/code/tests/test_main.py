"""Pruebas para 21-embodied-vlas-openvla-pi0-groot."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncode(unittest.TestCase):
    def test_image(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        tokens = main.encode_image_for_vla(img, embed_dim=4096)
        # 16*16 + 1 = 257
        self.assertEqual(tokens.shape, (257, 4096))

    def test_instruct(self):
        tokens = main.encode_instruct("pick up the block", embed_dim=4096)
        self.assertEqual(tokens.shape, (4, 4096))


class TestActionDecode(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal(4096)
        action = main.vla_action_decode(h, action_dim=7)
        self.assertEqual(action.shape, (7,))


class TestVLAForward(unittest.TestCase):
    def test_basic(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        action = main.vla_forward(img, "pick up the red block")
        self.assertEqual(action.shape, (7,))


class TestDiscretize(unittest.TestCase):
    def test_bins(self):
        actions = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.4])
        bins = main.discretize_actions(actions, n_bins=256)
        self.assertEqual(bins.shape, (7,))
        for b in bins:
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, 256)


class TestChunk(unittest.TestCase):
    def test_chunking(self):
        actions = np.arange(25)
        chunks = main.chunk_actions(actions, chunk_size=10)
        # 10, 10, 5
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].shape, (10,))
        self.assertEqual(chunks[2].shape, (5,))


class TestOpenVLA(unittest.TestCase):
    def test_forward(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        bins = main.openvla_forward(img, "pick up")
        self.assertEqual(bins.shape, (7,))


class TestPi0(unittest.TestCase):
    def test_flow(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        action = main.pi0_flow_matching(img, "pick up", n_steps=5, action_dim=7)
        self.assertEqual(action.shape, (7,))


class TestGR00T(unittest.TestCase):
    def test_simulation(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        action = main.groot_simulation(img, "pick up", action_dim=7)
        self.assertEqual(action.shape, (7,))


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