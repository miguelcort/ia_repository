"""Pruebas para 21-inferencia-de-texto-nli."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestNLI(unittest.TestCase):
    def test_entailment(self):
        # Hipotesis mas especifica que premisa -> entailment
        rel = main.nli_3way_mock("el gato come", "gato come")
        self.assertEqual(rel, 0)

    def test_neutral(self):
        rel = main.nli_3way_mock("el gato come", "el perro ladra")
        # 'el' en comun, no es subconjunto
        self.assertEqual(rel, 1)

    def test_contradiction(self):
        rel = main.nli_3way_mock("el gato come", "el gato duerme")
        # 'el', 'gato' en comun, pero 'duerme' != 'come'
        self.assertEqual(rel, 1)


class TestLoss(unittest.TestCase):
    def test_loss(self):
        y_true = np.array([1.0, 0.0, 0.0])
        logits = np.array([2.0, 1.0, 0.1])
        loss = main.contradiction_loss(y_true, logits)
        self.assertGreater(loss, 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Relacion", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()