"""Pruebas para 11-descomposicion-svd."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSVD(unittest.TestCase):
    def test_formas(self):
        A = np.random.default_rng(0).random((5, 3))
        U, S, Vt = main.svd(A)
        self.assertEqual(U.shape, (5, 3))
        self.assertEqual(S.shape, (3,))
        self.assertEqual(Vt.shape, (3, 3))

    def test_singular_values_ordenados(self):
        A = np.random.default_rng(0).random((5, 3))
        _, S, _ = main.svd(A)
        self.assertTrue(all(S[i] >= S[i + 1] for i in range(len(S) - 1)))

    def test_reconstruccion_full(self):
        A = np.random.default_rng(0).random((5, 3))
        U, S, Vt = main.svd(A)
        A_rec = main.reconstruccion(U, S, Vt, k=3)
        np.testing.assert_allclose(A_rec, A, atol=1e-10)

    def test_reconstruccion_truncada(self):
        A = np.random.default_rng(0).random((5, 3))
        U, S, Vt = main.svd(A)
        A_k = main.reconstruccion(U, S, Vt, k=1)
        self.assertEqual(A_k.shape, A.shape)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("U shape", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()