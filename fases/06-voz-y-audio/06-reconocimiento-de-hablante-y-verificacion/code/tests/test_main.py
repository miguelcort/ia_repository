"""Pruebas para 06-reconocimiento-de-hablante-y-verificacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEmbedding(unittest.TestCase):
    def test_shape(self):
        senal = np.random.default_rng(0).normal(size=16000)
        emb = main.embedding_hablante_mock(senal, dim=192)
        self.assertEqual(emb.shape, (192,))

    def test_determinista(self):
        senal = np.random.default_rng(0).normal(size=16000)
        e1 = main.embedding_hablante_mock(senal, dim=64)
        e2 = main.embedding_hablante_mock(senal, dim=64)
        np.testing.assert_array_equal(e1, e2)


class TestSimilitud(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 2.0])
        self.assertAlmostEqual(main.cosine_similarity(v, v), 1.0)


class TestVerificacion(unittest.TestCase):
    def test_match(self):
        v = np.array([1.0, 0.0])
        u = np.array([1.0, 0.0])
        self.assertEqual(main.verificacion_hablante(v, u, umbral=0.5)[0], 1)

    def test_no_match(self):
        v = np.array([1.0, 0.0])
        u = np.array([-1.0, 0.0])
        self.assertEqual(main.verificacion_hablante(v, u, umbral=0.5)[0], 0)


class TestIdentificacion(unittest.TestCase):
    def test_identifica(self):
        db = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        etiquetas = ["alice", "bob"]
        audio = np.array([1.0, 0.0])
        nombre, sim = main.identificacion_hablante(audio, db, etiquetas)
        self.assertEqual(nombre, "alice")


class TestEER(unittest.TestCase):
    def test_eer(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        positivos = [True, True, False, False]
        thr, eer = main.equal_error_rate(scores, positivos, positivos)
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Similitud", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()