"""Pruebas para 09-qwen-vl-family-dynamic-fps."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDynamicResolution(unittest.TestCase):
    def test_square(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        tiles = main.dynamic_resolution_tiles(img, max_tiles=4)
        self.assertEqual(tiles.shape[0], 1)

    def test_wide(self):
        img = np.random.default_rng(0).standard_normal((100, 400, 3))
        tiles = main.dynamic_resolution_tiles(img, max_tiles=4)
        # 1x4
        self.assertEqual(tiles.shape[0], 4)


class TestDynamicFPS(unittest.TestCase):
    def test_max_frames(self):
        # 48 frames @ 24fps = 2s, @ 2fps = 4 frames
        video = np.random.default_rng(0).standard_normal((48, 64, 64, 3))
        frames = main.dynamic_fps_sampling(video, target_fps=2.0, max_frames=8)
        self.assertEqual(frames.shape[0], 4)

    def test_caps_max(self):
        # long video -> capped
        video = np.random.default_rng(0).standard_normal((240, 64, 64, 3))  # 10s
        frames = main.dynamic_fps_sampling(video, target_fps=2.0, max_frames=8)
        self.assertEqual(frames.shape[0], 8)

    def test_short_video(self):
        video = np.random.default_rng(0).standard_normal((10, 64, 64, 3))
        frames = main.dynamic_fps_sampling(video, target_fps=2.0, max_frames=8)
        # 10/12 < 1, n=1
        self.assertGreaterEqual(frames.shape[0], 1)


class TestEncodeImage(unittest.TestCase):
    def test_shape(self):
        tiles = np.random.default_rng(0).standard_normal((4, 336, 336, 3))
        feats = main.encode_image_tiles(tiles, embed_dim=1280)
        self.assertEqual(feats.shape, (4, 577, 1280))


class TestEncodeVideo(unittest.TestCase):
    def test_shape(self):
        frames = np.random.default_rng(0).standard_normal((8, 224, 224, 3))
        feats = main.encode_video_frames(frames, embed_dim=1280)
        self.assertEqual(feats.shape, (8, 257, 1280))


class Test2DRoPE(unittest.TestCase):
    def test_shape(self):
        emb_h, emb_w = main.qwen2d_rope_position_emb(24, 24, 128)
        # dim=128 -> half=64
        self.assertEqual(emb_h.shape, (24, 64))
        self.assertEqual(emb_w.shape, (24, 64))


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