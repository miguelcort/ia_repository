"""Pruebas para 13-flow-matching-y-rectified-flows."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestInterpolation(unittest.TestCase):
    def test_t_cero(self):
        x0 = np.array([1.0, 2.0])
        x1 = np.array([3.0, 4.0])
        out = main.linear_interp(x0, x1, 0.0)
        np.testing.assert_array_equal(out, x0)

    def test_t_uno(self):
        x0 = np.array([1.0, 2.0])
        x1 = np.array([3.0, 4.0])
        out = main.linear_interp(x0, x1, 1.0)
        np.testing.assert_array_equal(out, x1)

    def test_t_punto_cinco(self):
        x0 = np.array([0.0])
        x1 = np.array([2.0])
        out = main.linear_interp(x0, x1, 0.5)
        self.assertEqual(out[0], 1.0)


class TestFlow(unittest.TestCase):
    def test_conditional_flow(self):
        x0 = np.zeros((1, 3))
        x1 = np.ones((1, 3))
        t = 0.7
        x_t, v = main.conditional_flow(x0, x1, t)
        # v = x1 - x0 = 1
        np.testing.assert_array_equal(v, x1 - x0)


class TestLoss(unittest.TestCase):
    def test_zero_si_perfecto(self):
        x0 = np.zeros((2, 4))
        x1 = np.ones((2, 4))
        v = x1 - x0
        loss = main.flow_matching_loss(v, x1, x0, t=0.5)
        self.assertEqual(loss, 0.0)


class TestSampleFlow(unittest.TestCase):
    def test_simple(self):
        # Si v = x1 - x0 constante, deberia llegar a x1
        x0 = np.array([[0.0, 0.0]])
        x1_target = np.array([[2.0, 3.0]])
        v = x1_target - x0
        def model_fn(x, t):
            return np.broadcast_to(v, x.shape)
        out = main.sample_flow(model_fn, x0, n_steps=10, dim=2)
        np.testing.assert_allclose(out, x1_target, atol=0.01)


class TestConcepts(unittest.TestCase):
    def test_flow_vs_diff(self):
        comps = main.flow_matching_vs_diffusion()
        self.assertIn("Flow matching (FM)", comps)
        self.assertIn("Rectified flow (RF)", comps)


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