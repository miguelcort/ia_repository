"""Pruebas para 03-clasificacion-de-audio."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestClassifier(unittest.TestCase):
    def test_shape(self):
        features = np.random.default_rng(0).normal(size=(10, 80))
        pesos = np.random.default_rng(1).normal(scale=0.1, size=(80, 5))
        bias = np.zeros(5)
        logits = main.mock_classifier(features, pesos, bias)
        self.assertEqual(logits.shape, (10, 5))


class TestLabel(unittest.TestCase):
    def test_label(self):
        logits = np.array([0.1, 0.5, 0.3])
        labels = ["a", "b", "c"]
        label, prob = main.label_clase(logits, labels)
        self.assertEqual(label, "b")


class TestAccuracy(unittest.TestCase):
    def test_perfecto(self):
        self.assertEqual(main.accuracy([0, 1, 2], [0, 1, 2]), 1.0)

    def test_cero(self):
        self.assertEqual(main.accuracy([0, 1, 2], [2, 0, 1]), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Accuracy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()