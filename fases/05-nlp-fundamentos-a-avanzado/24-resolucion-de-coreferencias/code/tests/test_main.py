"""Pruebas para 24-resolucion-de-coreferencias."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCoref(unittest.TestCase):
    def test_basico(self):
        ents = main.resolver_coreferencias_mock("Maria estudio. Ella comio.", ["Maria", "Ella"])
        self.assertEqual(len(ents), 2)
        self.assertEqual(ents[0]["mencion"], "Maria")


class TestDistancia(unittest.TestCase):
    def test_distancia(self):
        self.assertEqual(main.distancia_entre_menciones(0, 5), 5)


class TestSimilitud(unittest.TestCase):
    def test_similitud(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(4, 8))
        s = main.mention_pair_score(0, 1, emb)
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Entidades", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()