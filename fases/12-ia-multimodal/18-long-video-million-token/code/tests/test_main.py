"""Pruebas para 18-long-video-million-token."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKeyframes(unittest.TestCase):
    def test_basic(self):
        video = np.random.default_rng(0).standard_normal((1000, 32, 32, 3))
        kf = main.keyframe_extract(video, n_keyframes=16)
        self.assertEqual(kf.shape, (16, 32, 32, 3))


class TestHierarchical(unittest.TestCase):
    def test_levels(self):
        video = np.random.default_rng(0).standard_normal((32, 64, 64, 3))
        levels = main.hierarchical_compress(video, n_levels=3, factor=2)
        self.assertEqual(len(levels), 3)
        # each level smaller
        self.assertLess(levels[1].shape[0], levels[0].shape[0])
        self.assertLess(levels[2].shape[0], levels[1].shape[0])


class TestSlidingWindow(unittest.TestCase):
    def test_basic(self):
        seq = np.random.default_rng(0).standard_normal((1000, 8))
        windows = main.sliding_window_attention(seq, window_size=256, stride=128)
        # 0-256, 128-384, 256-512, 384-640, 512-768, 640-896, 768-1000
        self.assertGreater(len(windows), 1)
        # first window size 256
        self.assertEqual(windows[0].shape, (256, 8))


class TestRingAttention(unittest.TestCase):
    def test_rounds(self):
        rounds, ops = main.ring_attention_simulation(1000, n_devices=4, chunk_size=128)
        # 4 devices -> 6 rounds
        self.assertEqual(rounds, 6)
        # 8 chunks * 6 rounds = 48
        self.assertEqual(ops, 48)


class TestEncodeLong(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((1000, 32, 32, 3))
        tokens = main.encode_long_video(video, target_tokens=512, embed_dim=512)
        self.assertEqual(tokens.shape, (512, 512))


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