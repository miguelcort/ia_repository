"""Pruebas para 23-difusion-transformers-y-flujo-rectificado."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSchedule(unittest.TestCase):
    def test_linear(self):
        s = main.linear_schedule(100)
        self.assertEqual(s[0], 1e-4)
        self.assertAlmostEqual(s[-1], 0.02, places=5)

    def test_rectified(self):
        s = main.rectificado_flow_schedule(10)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(s[-1], 1.0)


class TestRectified(unittest.TestCase):
    def test_interpolacion(self):
        x0 = np.array([0.0])
        ruido = np.array([1.0])
        # En t=0 -> x0, t=1 -> ruido
        self.assertAlmostEqual(main.interpolacion_rectified(x0, ruido, 0)[0], 0.0)
        self.assertAlmostEqual(main.interpolacion_rectified(x0, ruido, 1)[0], 1.0)
        # En t=0.5 -> 0.5
        self.assertAlmostEqual(main.interpolacion_rectified(x0, ruido, 0.5)[0], 0.5)

    def test_velocidad(self):
        x0 = np.array([1.0])
        ruido = np.array([0.0])
        # velocidad = ruido - x0 = -1
        self.assertAlmostEqual(main.velocidad_target(x0, ruido)[0], -1.0)


class TestLoss(unittest.TestCase):
    def test_rectified_loss(self):
        v = np.array([1.0, 2.0])
        self.assertEqual(main.loss_rectified(v, v), 0.0)

    def test_ddpm_loss(self):
        e = np.array([0.5, 0.3])
        self.assertEqual(main.loss_ddpm(e, e), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Loss", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()