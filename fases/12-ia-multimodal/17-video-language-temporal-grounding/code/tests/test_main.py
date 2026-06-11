"""Pruebas para 17-video-language-temporal-grounding."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDenseFrames(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((100, 32, 32, 3))
        frames = main.sample_dense_frames(video, fps_sample=4, max_frames=16)
        # 100 / 4 = 25, cap 16
        self.assertEqual(frames.shape, (16, 32, 32, 3))

    def test_short_video(self):
        video = np.random.default_rng(0).standard_normal((10, 32, 32, 3))
        frames = main.sample_dense_frames(video, fps_sample=4, max_frames=16)
        # 10/4 = 2
        self.assertEqual(frames.shape, (2, 32, 32, 3))


class TestTemporalGrounding(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((100, 32, 32, 3))
        query = np.random.default_rng(1).standard_normal(512)
        moments = main.temporal_grounding(video, query, n_moments=3)
        self.assertEqual(len(moments), 3)
        for s, e in moments:
            self.assertGreaterEqual(s, 0)
            self.assertGreater(e, s)


class TestIoU(unittest.TestCase):
    def test_perfect_match(self):
        pred = [(0, 10)]
        gt = [(0, 10)]
        self.assertEqual(main.moment_retrieval_score(pred, gt, iou_threshold=0.5), 1)

    def test_no_overlap(self):
        pred = [(0, 5)]
        gt = [(20, 30)]
        self.assertEqual(main.moment_retrieval_score(pred, gt, iou_threshold=0.5), 0)

    def test_partial_overlap(self):
        pred = [(0, 10)]
        gt = [(5, 15)]
        # IoU = 5 / 15 = 0.33 < 0.5
        self.assertEqual(main.moment_retrieval_score(pred, gt, iou_threshold=0.5), 0)


class TestVideoTextSim(unittest.TestCase):
    def test_self_similarity(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 8))
        sim = main.video_text_similarity(v, v[0])
        # self row should give max sim
        # when video_emb[0] matches text_emb = video_emb[0], sim[0] = 1
        self.assertAlmostEqual(sim, 1.0, places=8)

    def test_orthogonal(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((5, 8))
        t = np.zeros(8)
        sim = main.video_text_similarity(v, t)
        # zero text -> 0 (norm=eps)
        self.assertAlmostEqual(sim, 0.0, places=8)


class TestEncodeClip(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((8, 32, 32, 3))
        emb = main.encode_video_clip(video, embed_dim=512)
        self.assertEqual(emb.shape, (8, 512))


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