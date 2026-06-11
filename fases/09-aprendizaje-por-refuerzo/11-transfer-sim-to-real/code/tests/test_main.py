"""Pruebas para 11-transfer-sim-to-real."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDomainRandomization(unittest.TestCase):
    def test_uniform(self):
        v = main.domain_randomization_param("test", "uniform", 0, 1, seed=0)
        self.assertGreaterEqual(v, 0)
        self.assertLessEqual(v, 1)

    def test_normal(self):
        v = main.domain_randomization_param("test", "normal", 0, 1, seed=0)
        # Mean 0.5, std 0.25
        self.assertGreater(v, -1)
        self.assertLess(v, 2)


class TestObsNoise(unittest.TestCase):
    def test_noise_shape(self):
        obs = np.array([1.0, 2.0, 3.0])
        noisy = main.add_observation_noise(obs, noise_std=0.1, seed=0)
        self.assertEqual(noisy.shape, obs.shape)

    def test_noise_zero(self):
        obs = np.array([1.0, 2.0, 3.0])
        noisy = main.add_observation_noise(obs, noise_std=0.0, seed=0)
        np.testing.assert_array_equal(noisy, obs)


class TestSystemID(unittest.TestCase):
    def test_residual(self):
        real = np.array([1.0, 2.0, 3.0])
        sim = np.array([0.9, 2.1, 2.95])
        residual = main.system_id_residual(real, sim)
        np.testing.assert_allclose(residual, [0.1, -0.1, 0.05], atol=1e-10)


class TestActionSpace(unittest.TestCase):
    def test_compatible(self):
        real = np.array([1.0, 2.0])
        sim = np.array([1.02, 1.99])
        self.assertTrue(main.action_space_check(real, sim))

    def test_incompatible(self):
        real = np.array([1.0, 2.0])
        sim = np.array([2.0, 1.0])
        self.assertFalse(main.action_space_check(real, sim))


class TestCategories(unittest.TestCase):
    def test_dynamics_randomization(self):
        c = main.dynamics_randomization()
        self.assertIn("Visual", c)
        self.assertIn("Dynamics", c)

    def test_mitigations(self):
        m = main.reality_gap_mitigations()
        self.assertIn("Domain randomization", m)
        self.assertIn("System identification", m)


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