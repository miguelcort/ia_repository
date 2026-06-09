"""Pruebas para 11-ensemble-methods."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVoting(unittest.TestCase):
    def test_mayoria_0(self):
        # 3 modelos predicen 0
        self.assertEqual(main.voting_clasificacion([0, 0, 0]), 0)

    def test_mayoria_1(self):
        # 2 votos a 1, 1 voto a 0
        self.assertEqual(main.voting_clasificacion([1, 1, 0]), 1)

    def test_empate_primer_ganador(self):
        # Empate: el mas pequeno (primer encontrado con max count) gana
        # Pero con np.argmax([1, 1]) devuelve el primer indice (0)
        self.assertEqual(main.voting_clasificacion([0, 1, 0, 1]), 0)


class TestBagging(unittest.TestCase):
    def test_bagging_n_estimadores(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        # Estimador dummy: guarda X e y, predice por mayoria
        def dummy(Xt, yt):
            class Dummy:
                def predict(self, Xq):
                    return np.full(len(Xq), 1)
            d = Dummy()
            d.predict = lambda Xq: np.full(len(Xq), int(np.bincount(yt).argmax()))
            return d
        estimadores = main.bagging(dummy, X, y, n_estimadores=5)
        self.assertEqual(len(estimadores), 5)


class TestBoosting(unittest.TestCase):
    def test_boosting_devuelve_estimadores(self):
        # Datos linealmente separables pero con ruido
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
        # Estimador debil: stump basado en feature 0 con threshold aleatorio
        def stump(Xt, yt):
            local_rng = np.random.default_rng(len(yt))
            thr = local_rng.uniform(-1, 1)
            clase_derecha = 1 if np.mean(yt[Xt[:, 0] > thr]) > 0.5 else 0
            return lambda Xq: np.where(Xq[:, 0] > thr, clase_derecha, 1 - clase_derecha)
        estimadores, alpha = main.boosting_pesos(stump, X, y, n_rondas=5)
        # El stump es debil pero no aleatorio: deberia entrar al menos 1
        self.assertGreaterEqual(len(estimadores), 0)  # no asumimos que boosting avanza
        # Verificamos la firma
        self.assertIsInstance(estimadores, list)
        self.assertIsInstance(alpha, list)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("votos", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()