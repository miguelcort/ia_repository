"""Pruebas para 08-llava-onevision-single-multi-video."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSampleFrames(unittest.TestCase):
    def test_uniform(self):
        video = np.random.default_rng(0).standard_normal((100, 32, 32, 3))
        frames = main.sample_video_frames(video, n_frames=8, strategy="uniform")
        self.assertEqual(frames.shape, (8, 32, 32, 3))

    def test_random(self):
        video = np.random.default_rng(0).standard_normal((100, 32, 32, 3))
        frames = main.sample_video_frames(video, n_frames=8, strategy="random")
        self.assertEqual(frames.shape, (8, 32, 32, 3))
        # indices sorted (sample_video_frames uses np.sort on indices)
        # each frame must be a real frame from video
        T = video.shape[0]
        for f in frames:
            # find matching frame in video
            found = any(np.array_equal(f, v) for v in video)
            self.assertTrue(found)


class TestOneVisionTileImage(unittest.TestCase):
    def test_square(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        tiles = main.onevision_tile_image(img, max_tiles=4)
        # 1x1
        self.assertEqual(tiles.shape[0], 1)

    def test_wide(self):
        img = np.random.default_rng(0).standard_normal((100, 400, 3))
        tiles = main.onevision_tile_image(img, max_tiles=4)
        # 1x4
        self.assertEqual(tiles.shape[0], 4)

    def test_tall(self):
        img = np.random.default_rng(0).standard_normal((400, 100, 3))
        tiles = main.onevision_tile_image(img, max_tiles=4)
        # 4x1
        self.assertEqual(tiles.shape[0], 4)


class TestOneVisionTileVideo(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((30, 100, 100, 3))
        tiles = main.onevision_tile_video(video, n_frames=4, max_tiles_per_frame=1)
        # 4 frames * 1 tile = 4
        self.assertEqual(tiles.shape[0], 4)


class TestEncodeTiles(unittest.TestCase):
    def test_shape(self):
        tiles = np.random.default_rng(0).standard_normal((4, 336, 336, 3))
        feats = main.encode_tiles(tiles, embed_dim=1024)
        # 4 tiles, 24*24+1 = 577 patches, 1024 dim
        self.assertEqual(feats.shape, (4, 577, 1024))


class TestOneVisionForward(unittest.TestCase):
    def test_image_only(self):
        img = np.random.default_rng(0).standard_normal((672, 1008, 3))
        out = main.onevision_forward(image=img, embed_dim=1024)
        self.assertEqual(out.shape[1], 1024)
        self.assertGreater(out.shape[0], 0)

    def test_video_only(self):
        video = np.random.default_rng(0).standard_normal((30, 224, 224, 3))
        out = main.onevision_forward(video=video, embed_dim=1024)
        self.assertEqual(out.shape[1], 1024)

    def test_image_plus_video(self):
        img = np.random.default_rng(0).standard_normal((400, 400, 3))
        video = np.random.default_rng(1).standard_normal((30, 224, 224, 3))
        out = main.onevision_forward(image=img, video=video, embed_dim=1024)
        self.assertEqual(out.shape[1], 1024)

    def test_none(self):
        out = main.onevision_forward(embed_dim=1024)
        self.assertIsNone(out)


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