"""Pruebas para 09-teoria-de-la-informacion."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestEntropia(unittest.TestCase):
    def test_moneda_justa(self):
        self.assertAlmostEqual(main.entropia(np.array([0.5, 0.5])), 1.0)

    def test_evento_cierto(self):
        # P(X=x) = 1 -> H = 0
        self.assertAlmostEqual(main.entropia(np.array([1.0, 0.0])), 0.0)

    def test_moneda_sesgada(self):
        # H(0.9, 0.1) = -0.9 log2 0.9 - 0.1 log2 0.1 ~ 0.469
        h = main.entropia(np.array([0.9, 0.1]))
        self.assertAlmostEqual(h, 0.4690, places=3)

    def test_entropia_maxima(self):
        # Uniforme sobre n elementos: H = log2(n)
        p = np.ones(8) / 8
        self.assertAlmostEqual(main.entropia(p), 3.0)


class TestEntropiaConjunta(unittest.TestCase):
    def test_independientes(self):
        # Si X y Y son independientes, H(X,Y) = H(X) + H(Y)
        p_xy = np.array([[0.25, 0.25], [0.25, 0.25]])
        h_x = main.entropia(np.array([0.5, 0.5]))
        h_y = main.entropia(np.array([0.5, 0.5]))
        self.assertAlmostEqual(main.entropia_conjunta(p_xy), h_x + h_y)


class TestInformacionMutua(unittest.TestCase):
    def test_independientes_imi_cero(self):
        p_xy = np.array([[0.25, 0.25], [0.25, 0.25]])
        self.assertAlmostEqual(main.informacion_mutua(p_xy), 0.0, places=5)

    def test_identicas_max(self):
        # X = Y: p(x,y) = 1 si x==y
        p = np.eye(2) / 2
        # I(X;X) = H(X) = 1
        self.assertAlmostEqual(main.informacion_mutua(p), 1.0, places=5)


class TestDivergenciaKL(unittest.TestCase):
    def test_kl_de_p_a_p_es_cero(self):
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(main.divergencia_kl(p, p), 0.0)

    def test_kl_no_negativa(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        self.assertGreaterEqual(main.divergencia_kl(p, q), 0.0)


class TestEntropiaCruzada(unittest.TestCase):
    def test_iguales_es_entropia(self):
        p = np.array([0.5, 0.5])
        h = main.entropia(p)
        h_cross = main.entropia_cruzada(p, p)
        self.assertAlmostEqual(h, h_cross)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Entropia", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
