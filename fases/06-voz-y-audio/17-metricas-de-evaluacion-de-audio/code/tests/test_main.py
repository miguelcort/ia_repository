"""Pruebas para 17-metricas-de-evaluacion-de-audio."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPESQ(unittest.TestCase):
    def test_identidad(self):
        audio = np.random.default_rng(0).normal(size=16000)
        score = main.pesq_score(audio, audio)
        self.assertGreaterEqual(score, 3.5)
        self.assertLessEqual(score, 4.5)

    def test_rango(self):
        a = np.random.default_rng(0).normal(size=16000)
        b = np.random.default_rng(1).normal(size=16000)
        score = main.pesq_score(a, b)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 4.5)


class TestSTOI(unittest.TestCase):
    def test_identidad(self):
        audio = np.random.default_rng(0).normal(size=16000)
        score = main.stoi_score(audio, audio)
        self.assertGreater(score, 0.9)

    def test_rango(self):
        self.assertGreaterEqual(main.stoi_score(np.zeros(1000), np.zeros(1000)), 0.0)
        self.assertLessEqual(main.stoi_score(np.zeros(1000), np.zeros(1000)), 1.0)


class TestViSQOL(unittest.TestCase):
    def test_rango(self):
        audio = np.random.default_rng(0).normal(size=16000)
        self.assertGreaterEqual(main.visqol_score(audio, audio), 1.0)
        self.assertLessEqual(main.visqol_score(audio, audio), 5.0)


class TestMOS(unittest.TestCase):
    def test_rango(self):
        audio = np.random.default_rng(0).normal(size=16000)
        self.assertGreaterEqual(main.mos_estimate(audio, audio), 1.0)
        self.assertLessEqual(main.mos_estimate(audio, audio), 5.0)


class TestFAD(unittest.TestCase):
    def test_basico(self):
        a = np.random.default_rng(0).normal(size=(10, 32))
        b = np.random.default_rng(1).normal(size=(10, 32))
        score = main.fad_score(a, b)
        self.assertGreaterEqual(score, 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("PESQ", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()