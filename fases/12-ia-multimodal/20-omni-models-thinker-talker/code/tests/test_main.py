"""Pruebas para 20-omni-models-thinker-talker."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestThinker(unittest.TestCase):
    def test_text(self):
        h = main.omni_thinker([("text", "hello world")])
        self.assertEqual(h.shape, (2, 4096))

    def test_audio(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000)
        h = main.omni_thinker([("audio", audio)])
        # 16000 / 320 = 50
        self.assertEqual(h.shape, (50, 4096))

    def test_image(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((64, 64, 3))
        h = main.omni_thinker([("image", img)])
        # 4*4 + 1 = 17
        self.assertEqual(h.shape, (17, 4096))

    def test_multimodal(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((64, 64, 3))
        audio = rng.standard_normal(16000)
        modalities = [("text", "hi"), ("audio", audio), ("image", img)]
        h = main.omni_thinker(modalities)
        # 1 + 50 + 17 = 68
        self.assertEqual(h.shape, (68, 4096))


class TestTalker(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal((10, 4096))
        text_logits, audio_codes = main.omni_talker(h, vocab_size=1000, audio_codebook=100)
        self.assertEqual(text_logits.shape, (10, 1000))
        self.assertEqual(audio_codes.shape, (10,))
        self.assertLess(audio_codes.max(), 100)


class TestStreaming(unittest.TestCase):
    def test_first_token(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal((10, 4096))
        out = main.omni_streaming_first_token(h, latency_target_ms=200)
        self.assertIn("latency_ms", out)
        self.assertLessEqual(out["latency_ms"], 200)


class TestGPT4oForward(unittest.TestCase):
    def test_text(self):
        h_ids = main.gpt4o_style_forward([("text", "hello")], target_modality="text")
        self.assertEqual(h_ids.shape, (1,))

    def test_audio(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000)
        codes = main.gpt4o_style_forward([("audio", audio)], target_modality="audio")
        self.assertEqual(codes.shape, (50,))


class TestQwenOmni(unittest.TestCase):
    def test_units(self):
        v, a = main.qwen_omni_speech_unit()
        self.assertEqual(v, 200000)
        self.assertEqual(a, 4096)


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