"""Pruebas para 04-clasificacion-de-imagenes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSoftmax(unittest.TestCase):
    def test_suma_uno(self):
        z = np.array([1.0, 2.0, 3.0])
        s = main.softmax(z)
        self.assertAlmostEqual(s.sum(), 1.0)

    def test_dominante(self):
        z = np.array([0.0, 10.0])
        s = main.softmax(z)
        # El segundo deberia dominar
        self.assertGreater(s[1], 0.99)


class TestLoss(unittest.TestCase):
    def test_perfecto(self):
        # Logits muy altos en la clase correcta
        logits = np.array([[10.0, 0.0, 0.0]])
        y = np.array([[1, 0, 0]])
        loss = main.cross_entropy(logits, y)
        self.assertLess(loss, 0.01)

    def test_incorrecto(self):
        logits = np.array([[0.0, 10.0, 0.0]])
        y = np.array([[1, 0, 0]])
        loss = main.cross_entropy(logits, y)
        self.assertGreater(loss, 5.0)


class TestTopK(unittest.TestCase):
    def test_top1(self):
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        y = np.array([2, 0])
        self.assertEqual(main.top_k_accuracy(logits, y, k=1), 1.0)

    def test_top2(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        y = np.array([2])
        self.assertEqual(main.top_k_accuracy(logits, y, k=1), 1.0)
        # Top-2 con clase verdadera 2 -> true
        self.assertEqual(main.top_k_accuracy(logits, y, k=2), 1.0)


class TestConfusion(unittest.TestCase):
    def test_simple(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])  # un error: t=1,p=0
        cm = main.confusion_matriz(y_true, y_pred, n_clases=3)
        # Diagonal: 2, 1, 2. Error: y_true=1 -> y_pred=0
        self.assertEqual(cm[0, 0], 2)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[2, 2], 2)
        # M[1, 0] = 1 (true=1, pred=0)
        self.assertEqual(cm[1, 0], 1)


class TestF1(unittest.TestCase):
    def test_perfecto(self):
        cm = np.array([[10, 0], [0, 10]])
        self.assertAlmostEqual(main.f1_macro(cm), 1.0)


class TestAugment(unittest.TestCase):
    def test_flip(self):
        img = np.array([[1, 2], [3, 4]])
        flipped = main.augmentar_flip_horizontal(img, semilla=0)
        # A veces se flipea, a veces no. Verificamos shape
        self.assertEqual(flipped.shape, (2, 2))

    def test_crop(self):
        img = np.arange(9, dtype=float).reshape(3, 3)
        out = main.augmentar_random_crop(img, pad=2)
        self.assertEqual(out.shape, (3, 3))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Cross-entropy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()