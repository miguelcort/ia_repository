"""Pruebas para 14-deteccion-de-actividad-de-voz-y-turn-taking."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVADEnergia(unittest.TestCase):
    def test_habla(self):
        audio = np.random.default_rng(0).normal(scale=0.1, size=16000)
        segs = main.vad_energia(audio, threshold=0.01)
        self.assertGreater(len(segs), 0)

    def test_silencio(self):
        audio = np.zeros(16000)
        segs = main.vad_energia(audio, threshold=0.01)
        self.assertEqual(len(segs), 0)

    def test_dos_segmentos(self):
        sr = 16000
        audio = np.concatenate([
            np.random.default_rng(0).normal(scale=0.1, size=sr),
            np.zeros(sr // 2),
            np.random.default_rng(1).normal(scale=0.1, size=sr),
        ])
        segs = main.vad_energia(audio, sample_rate=sr, threshold=0.01)
        self.assertGreaterEqual(len(segs), 2)


class TestSileroVAD(unittest.TestCase):
    def test_basico(self):
        audio = np.random.default_rng(0).normal(scale=0.1, size=16000)
        segs = main.vad_silero_mock(audio)
        self.assertGreater(len(segs), 0)


class TestEndpointing(unittest.TestCase):
    def test_merge(self):
        sr = 16000
        # Dos segmentos con gap corto (merge) y gap largo (no merge)
        segs = [
            (True, 0, int(0.5 * sr)),
            (True, int(0.6 * sr), int(1.0 * sr)),  # gap 100ms < 500ms
            (True, int(2.0 * sr), int(2.5 * sr)),  # gap 1s > 500ms
        ]
        merged = main.endpointing_por_silencio(segs, min_silence_ms=500, sample_rate=sr)
        # Primer y segundo merge -> 1, tercero separado -> 2 total
        self.assertEqual(len(merged), 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Segmentos", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()