"""Pruebas para 02-bolsa-de-palabras-y-tfidf."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVocab(unittest.TestCase):
    def test_construir(self):
        docs = ["el gato come", "el perro ladra"]
        v = main.construir_vocabulario(docs)
        # Palabras con freq >= 1
        self.assertIn("el", v)
        self.assertIn("gato", v)
        self.assertIn("perro", v)
        self.assertIn("come", v)
        self.assertIn("ladra", v)
        # El vocabulario es un dict palabra -> indice
        self.assertEqual(len(v), 5)

    def test_min_freq(self):
        docs = ["el gato come", "el perro come"]
        v = main.construir_vocabulario(docs, min_freq=2)
        # "el" aparece en 2 docs, "gato" en 1
        self.assertIn("el", v)
        self.assertNotIn("gato", v)


class TestBoW(unittest.TestCase):
    def test_basico(self):
        docs = ["el gato come"]
        v = main.construir_vocabulario(docs)
        bow = main.bag_of_words(docs[0], v)
        self.assertEqual(bow.sum(), 3)  # 3 tokens
        # "el" aparece 1 vez
        self.assertEqual(bow[v["el"]], 1)
        # "gato" aparece 1 vez
        self.assertEqual(bow[v["gato"]], 1)

    def test_vocab_desconocido(self):
        v = {"a": 0, "b": 1}
        bow = main.bag_of_words("c d e", v)
        np.testing.assert_array_equal(bow, np.zeros(2))


class TestTFIDF(unittest.TestCase):
    def test_shape(self):
        docs = ["el gato come", "el perro ladra"]
        v = main.construir_vocabulario(docs)
        mat = main.tf_idf(docs, v)
        self.assertEqual(mat.shape, (2, len(v)))

    def test_idf_comun(self):
        # "el" en todos los docs -> IDF bajo
        docs = ["el gato", "el perro", "el pajaro"]
        v = main.construir_vocabulario(docs)
        mat = main.tf_idf(docs, v)
        # El valor de "el" deberia ser menor que "gato" (que aparece solo en 1)
        if "el" in v and "gato" in v:
            self.assertLess(mat[0, v["el"]], mat[0, v["gato"]])


class TestSimilarity(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(main.cosine_similarity(v, v), 1.0)

    def test_ortogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(main.cosine_similarity(a, b), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("TF-IDF", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()