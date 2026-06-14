"""Pruebas para la lección 09 — Inpainting, outpainting y edición."""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


class TestSinEmbed(unittest.TestCase):
    def test_dim_correcta(self):
        emb = main.sin_embed(0, 100, dim=8)
        self.assertEqual(len(emb), 8)

    def test_valores_acotados(self):
        for t in (0, 10, 100, 1000):
            emb = main.sin_embed(t, 100, dim=16)
            for v in emb:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)


class TestMathHelpers(unittest.TestCase):
    def test_tanh(self):
        self.assertAlmostEqual(main.tanh([0.0])[0], 0.0)
        self.assertAlmostEqual(main.tanh([1.0])[0], math.tanh(1.0))

    def test_tanh_grad(self):
        self.assertAlmostEqual(main.tanh_grad([0.0])[0], 1.0)
        self.assertAlmostEqual(main.tanh_grad([1.0])[0], 0.0, places=6)


class TestSchedule(unittest.TestCase):
    def test_alphas_en_rango(self):
        alphas, _ = main.make_schedule(50)
        for a in alphas:
            self.assertGreater(a, 0.0)
            self.assertLess(a, 1.0)

    def test_alpha_bars_decrecientes(self):
        _, bars = main.make_schedule(50)
        for i in range(1, len(bars)):
            self.assertLess(bars[i], bars[i - 1])


class TestSampleData(unittest.TestCase):
    def test_data_shape(self):
        rng = random.Random(0)
        data, cluster = main.sample_data(rng, d=5)
        self.assertEqual(len(data), 5)
        self.assertIn(cluster, (0, 1))

    def test_data_alrededor_de_centro(self):
        rng = random.Random(0)
        # muestrea 100 veces, debería estar cerca de -1 o +1
        for _ in range(50):
            data, cluster = main.sample_data(rng, d=5)
            center = -1.0 if cluster == 0 else 1.0
            for v in data:
                self.assertLess(abs(v - center), 2.0)


class TestNetwork(unittest.TestCase):
    def test_forward_shape(self):
        rng = random.Random(0)
        net = main.init_net(x_dim=5, t_dim=8, hidden=16, rng=rng)
        out, cache = main.forward([0.1] * 5, [0.0] * 8, net)
        self.assertEqual(len(out), 5)
        self.assertIn("h1", cache)
        self.assertIn("h2", cache)


class TestInpaint(unittest.TestCase):
    def test_inpaint_respeta_pined_dims(self):
        rng = random.Random(0)
        T, t_dim, hidden, d = 20, 8, 16, 5
        alphas, alpha_bars = main.make_schedule(T)
        net = main.init_net(d, t_dim, hidden, rng)
        main.train(
            net, alpha_bars, T, steps=500, lr=0.01,
            t_dim=t_dim, d=d, rng=rng,
        )
        clean = [1.0, 1.0, 1.0, -1.0, -1.0]
        mask = [False, False, False, True, True]  # pinea 0-2
        out = main.inpaint(
            net, alphas, alpha_bars, T, t_dim, d, clean, mask, rng
        )
        # Las dims pineadas deben quedar exactamente como clean
        for i in range(3):
            self.assertAlmostEqual(out[i], clean[i], places=5)

    def test_inpaint_genera_dims_enmascaradas(self):
        rng = random.Random(0)
        T, t_dim, hidden, d = 20, 8, 16, 5
        alphas, alpha_bars = main.make_schedule(T)
        net = main.init_net(d, t_dim, hidden, rng)
        main.train(
            net, alpha_bars, T, steps=500, lr=0.01,
            t_dim=t_dim, d=d, rng=rng,
        )
        clean = [1.0, 1.0, 1.0, 0.0, 0.0]
        mask = [False, False, False, True, True]
        out = main.inpaint(
            net, alphas, alpha_bars, T, t_dim, d, clean, mask, rng
        )
        # Las dims regeneradas no son None y son números reales
        for i in range(3, 5):
            self.assertIsInstance(out[i], float)
            self.assertFalse(math.isnan(out[i]))


class TestMain(unittest.TestCase):
    def test_main_retorna_cero(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        self.assertIn("inpainting", salida)
        self.assertIn("outpainting", salida)


if __name__ == "__main__":
    unittest.main()
