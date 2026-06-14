"""Pruebas para la lección 02 — Reward hacking y Goodhart."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


class TestGoldReward(unittest.TestCase):
    def test_producto_punto_basico(self):
        w = [1.0, 2.0, 3.0]
        x = [1.0, 1.0, 1.0]
        self.assertAlmostEqual(main.gold_reward(w, x), 6.0)

    def test_ponderado_negativo(self):
        w = [1.0, -1.0]
        x = [0.5, 0.5]
        self.assertAlmostEqual(main.gold_reward(w, x), 0.0)


class TestFitProxyRM(unittest.TestCase):
    def test_recupera_pesos_sin_ruido(self):
        # features con varianza suficiente para que la matriz sea
        # invertible
        import math
        random_features = [
            [
                math.cos(0.3 * i) + 0.1 * i,
                math.sin(0.3 * i) - 0.1 * i,
                0.5 * i + 0.1 * math.cos(i),
            ]
            for i in range(-5, 6)
        ]
        true_w = [1.0, -0.5, 0.3]
        w = main.fit_proxy_rm(true_w, random_features, noise_scale=0.0)
        for wi, wj in zip(w, true_w):
            self.assertAlmostEqual(wi, wj, places=1)


class TestSolveLinear(unittest.TestCase):
    def test_sistema_2x2(self):
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [4.0, 7.0]
        x = main._solve_linear(A, b)
        # x1=1, x2=2
        self.assertAlmostEqual(x[0], 1.0, places=4)
        self.assertAlmostEqual(x[1], 2.0, places=4)


class TestKLDivergence(unittest.TestCase):
    def test_kl_de_p_a_p_es_cero(self):
        mu = [0.0, 0.0]
        kl = main.kl_divergence_gaussians(mu, 1.0, mu, 1.0)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_aumenta_con_diferencia(self):
        mu1 = [0.0]
        mu2 = [2.0]
        kl = main.kl_divergence_gaussians(mu1, 1.0, mu2, 1.0)
        self.assertGreater(kl, 1.0)


class TestHillClimb(unittest.TestCase):
    def test_hill_climb_mejora_proxy(self):
        proxy_w = [1.0, 1.0, 1.0]
        initial = [0.0, 0.0, 0.0]
        traj = main.hill_climb(
            initial_features=initial,
            proxy_weights=proxy_w,
            initial_weights=proxy_w,
            n_steps=30,
            beta=0.0,  # sin KL, debe subir sin restricción
        )
        proxy_inicial = traj[0][1]
        proxy_final = traj[-1][1]
        self.assertGreater(proxy_final, proxy_inicial)

    def test_kl_crece_sin_penalty(self):
        proxy_w = [1.0, 1.0, 1.0]
        initial = [0.0, 0.0, 0.0]
        traj = main.hill_climb(
            initial_features=initial,
            proxy_weights=proxy_w,
            initial_weights=proxy_w,
            n_steps=30,
            beta=0.0,
        )
        kl_inicial = traj[0][0]
        kl_final = traj[-1][0]
        self.assertGreater(kl_final, kl_inicial)


class TestSampleStudentT(unittest.TestCase):
    def test_sample_t_retorna_finito(self):
        for _ in range(50):
            sample = main._sample_student_t(5.0)
            self.assertTrue(isinstance(sample, float))
            self.assertTrue(abs(sample) < 100)


class TestMain(unittest.TestCase):
    def test_main_retorna_cero(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        self.assertIn("goodhart", salida.lower())
        self.assertIn("Catastrophic", salida)


if __name__ == "__main__":
    unittest.main()
