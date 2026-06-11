"""Pruebas para 14-evaluacion-fid-y-clip-score."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestFrechet(unittest.TestCase):
    def test_zero_si_identico(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((50, 8))
        # Si real == fake, FID = 0
        fid = main.fid_score(f, f)
        self.assertLess(fid, 0.1)

    def test_positivo(self):
        rng = np.random.default_rng(0)
        real = rng.standard_normal((50, 8))
        fake = rng.standard_normal((50, 8)) + 0.5
        fid = main.fid_score(real, fake)
        self.assertGreater(fid, 0)


class TestIS(unittest.TestCase):
    def test_alta_con_distribuciones_picudas(self):
        # Si cada sample tiene una clase dominante
        logits = np.zeros((10, 5))
        for i in range(10):
            logits[i, i % 5] = 10
        is_score = main.inception_score(logits)
        # IS alto cuando p(y|x) picuda y p(y) uniforme
        self.assertGreater(is_score, 4.5)


class TestCLIPScore(unittest.TestCase):
    def test_alta_cuando_alineados(self):
        # Imagen y texto con mismos features
        img = np.random.default_rng(0).standard_normal((4, 8))
        txt = img.copy()
        score = main.clip_score(img, txt)
        # Cosine sim de vectores identicos = 1
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_baja_cuando_ortogonales(self):
        img = np.zeros((1, 4))
        img[0, 0] = 1
        txt = np.zeros((1, 4))
        txt[0, 1] = 1
        score = main.clip_score(img, txt)
        # Cosine sim de vectores ortogonales = 0
        self.assertAlmostEqual(score, 0.0, places=4)


class TestPrecisionRecall(unittest.TestCase):
    def test_perfect(self):
        # Si real y fake identicos, precision = recall = 1
        rng = np.random.default_rng(0)
        f = rng.standard_normal((20, 4))
        p, r = main.precision_recall(f, f, k=2)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)


class TestSummary(unittest.TestCase):
    def test_metricas_presentes(self):
        s = main.metric_summary()
        self.assertIn("FID", s)
        self.assertIn("CLIP score", s)


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