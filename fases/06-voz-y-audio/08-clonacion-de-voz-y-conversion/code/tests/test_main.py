"""Pruebas para 08-clonacion-de-voz-y-conversion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSpeakerEncoder(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).normal(size=16000)
        emb = main.speaker_encoder_mock(audio, dim=192)
        self.assertEqual(emb.shape, (192,))


class TestVoiceConversion(unittest.TestCase):
    def test_shape(self):
        source = np.random.default_rng(0).normal(size=16000)
        target_emb = main.speaker_encoder_mock(np.random.default_rng(1).normal(size=16000))
        converted = main.voice_conversion_mock(source, target_emb)
        self.assertEqual(converted.shape, source.shape)

    def test_truncado(self):
        source = np.random.default_rng(0).normal(size=16000)
        target_emb = main.speaker_encoder_mock(np.random.default_rng(1).normal(size=16000))
        converted = main.voice_conversion_mock(source, target_emb, output_dim=1000)
        self.assertEqual(converted.shape, (1000,))


class TestSimilarity(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 0.0])
        self.assertAlmostEqual(main.speaker_similarity(v, v), 1.0)

    def test_ortogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(main.speaker_similarity(a, b), 0.0)


class TestConsistency(unittest.TestCase):
    def test_promedio(self):
        audios = [np.random.default_rng(i).normal(size=16000) for i in range(3)]
        ref = main.speaker_encoder_mock(audios[0])
        score = main.speaker_consistency(audios[1:], ref)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("sim", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()