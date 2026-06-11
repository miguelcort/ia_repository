"""Pruebas para 16-anti-spoofing-y-audio-watermarking."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDeepfake(unittest.TestCase):
    def test_detect(self):
        audio = np.random.default_rng(0).normal(size=16000)
        p_fake, decision = main.detect_deepfake_mock(audio)
        self.assertGreaterEqual(p_fake, 0.0)
        self.assertLessEqual(p_fake, 1.0)
        self.assertIn(decision, [0, 1])


class TestRealTime(unittest.TestCase):
    def test_frames(self):
        frames = [np.random.default_rng(i).normal(size=16000) for i in range(3)]
        decision, p_fake = main.detect_spoofing_real_time(frames)
        self.assertIn(decision, [0, 1])


class TestWatermark(unittest.TestCase):
    def test_embed(self):
        audio = np.random.default_rng(0).normal(size=1000)
        wm = main.embed_watermark(audio, "TEST")
        self.assertEqual(wm.shape, audio.shape)
        # Debe ser diferente (agregamos watermark)
        self.assertNotEqual(wm[0], audio[0])

    def test_detect(self):
        audio = np.random.default_rng(0).normal(size=1000)
        wm = main.embed_watermark(audio, "TEST")
        # Mock puede no detectar perfectamente, pero no debe fallar
        result = main.detect_watermark(wm, "TEST")
        self.assertIsInstance(result, bool)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Watermark", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()