"""Pruebas para 07-regularizacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDropout(unittest.TestCase):
    def test_inactivos(self):
        x = np.ones((100, 10))
        out, mask = main.dropout_forward(x, p=0.5, semilla=0)
        # ~50% deberian ser 0. Para 1000 elementos, ~500 inactivos
        inactivos = np.sum(out == 0)
        self.assertGreater(inactivos, 400)
        self.assertLess(inactivos, 600)

    def test_inference(self):
        x = np.ones((10, 5))
        out, mask = main.dropout_forward(x, p=0.5, entrenamiento=False)
        # En inference no se aplica
        np.testing.assert_array_equal(out, x)
        self.assertIsNone(mask)

    def test_escalado(self):
        # Inverted dropout: la salida esperada deberia mantener magnitud
        x = np.ones((1000, 100))
        out, _ = main.dropout_forward(x, p=0.5, semilla=0)
        # Media esperada: x * 1 (porque escala por 1/(1-p)=2)
        self.assertAlmostEqual(out.mean(), 1.0, places=1)


class TestL2(unittest.TestCase):
    def test_cero_para_ceros(self):
        W = np.zeros((3, 3))
        self.assertEqual(main.l2_pesos(W), 0.0)

    def test_grad(self):
        W = np.array([[1.0, -2.0]])
        g = main.l2_grad(W, lambda_=0.1)
        # 2 * 0.1 * 1.0 = 0.2
        self.assertAlmostEqual(g[0, 0], 0.2)
        # 2 * 0.1 * -2.0 = -0.4
        self.assertAlmostEqual(g[0, 1], -0.4)


class TestL1(unittest.TestCase):
    def test_norma_l1(self):
        W = np.array([[3.0, -4.0]])
        self.assertAlmostEqual(main.l1_pesos(W, lambda_=1.0), 7.0)

    def test_grad(self):
        W = np.array([[1.0, -2.0, 0.0]])
        g = main.l1_grad(W, lambda_=0.1)
        np.testing.assert_array_equal(g, [[0.1, -0.1, 0.0]])


class TestBatchNorm(unittest.TestCase):
    def test_media_cero(self):
        x = np.random.default_rng(0).normal(size=(100, 5))
        out, _, _, _ = main.batch_norm_forward(x)
        self.assertTrue(np.allclose(out.mean(axis=0), 0.0, atol=1e-6))

    def test_std_uno(self):
        x = np.random.default_rng(0).normal(size=(100, 5))
        out, _, _, _ = main.batch_norm_forward(x)
        self.assertTrue(np.allclose(out.std(axis=0), 1.0, atol=1e-3))

    def test_inference_sin_moviles(self):
        x = np.random.default_rng(0).normal(size=(50, 3))
        out, _, _, _ = main.batch_norm_forward(x, entrenamiento=False, media_movil=None, var_movil=None)
        # Sin estadisticas moviles, usa las del batch actual
        self.assertEqual(out.shape, x.shape)


class TestEarlyStopping(unittest.TestCase):
    def test_no_para_temprano(self):
        # Solo 2 epocas, paciencia=3: no debe parar
        self.assertFalse(main.early_stopping([0.5, 0.4], paciencia=3))

    def test_para_si_no_mejora(self):
        # 0.4 es mejor, las ultimas 3 no mejoran
        self.assertTrue(main.early_stopping([0.5, 0.4, 0.45, 0.46, 0.47], paciencia=3))

    def test_no_para_si_mejora(self):
        self.assertFalse(main.early_stopping([0.5, 0.4, 0.3, 0.2, 0.1], paciencia=3))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Dropout", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()