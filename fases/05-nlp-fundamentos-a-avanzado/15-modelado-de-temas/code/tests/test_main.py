"""Pruebas para 15-modelado-de-temas."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLDA(unittest.TestCase):
    def test_shape(self):
        docs = ["el gato come pescado", "el perro come carne"]
        theta, phi, vocab = main.lda_gibbs_step(docs, k_topics=2, n_iter=10, semilla=42)
        self.assertEqual(theta.shape, (2, 2))  # 2 docs, 2 topics
        self.assertEqual(phi.shape, (2, len(vocab)))
        # Suma 1 por doc
        for d in range(2):
            self.assertAlmostEqual(float(theta[d].sum()), 1.0, places=5)
        # Suma 1 por topic
        for k in range(2):
            self.assertAlmostEqual(float(phi[k].sum()), 1.0, places=5)

    def test_vocab(self):
        docs = ["el gato come"]
        _, _, vocab = main.lda_gibbs_step(docs, k_topics=1, n_iter=5, semilla=0)
        self.assertIn("gato", vocab)
        self.assertIn("come", vocab)
        self.assertIn("el", vocab)


class TestTopPalabras(unittest.TestCase):
    def test_top(self):
        vocab = ["a", "b", "c"]
        phi = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
        top = main.top_palabras_por_topic(phi, vocab, top_n=2)
        # Topic 0: 'c' (0.7), 'b' (0.2)
        self.assertEqual(top[0][0], "c")
        # Topic 1: 'a' (0.5), 'b' (0.3)
        self.assertEqual(top[1][0], "a")


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Topic", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()