"""Pruebas para 06-bert-masked-language-modeling."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMaskTokens(unittest.TestCase):
    def test_shape(self):
        tokens = [101, 2024, 1996, 3006, 102, 3231]
        masked, labels = main.mask_tokens(tokens, 103, 30000, seed=0)
        self.assertEqual(len(masked), len(tokens))
        self.assertEqual(len(labels), len(tokens))

    def test_especiales_no_maskeados(self):
        # CLS (0) y SEP (1) nunca se maskear
        tokens = [0, 1, 5, 6, 1, 7, 8, 1]
        for seed in range(10):
            masked, labels = main.mask_tokens(tokens, 103, 30000, seed=seed)
            # Posiciones 0 y 1 (CLS) no deberian tener label
            self.assertEqual(labels[0], -100)
            self.assertEqual(labels[1], -100)

    def test_labels_coinciden_con_original(self):
        # Donde label != -100, deberia ser el token original
        tokens = [101, 2024, 1996, 3006, 102, 3231]
        masked, labels = main.mask_tokens(tokens, 103, 30000, mlm_prob=1.0, seed=42)
        # Con prob 1.0, todos deberian tener label (excepto specials)
        non_special = [i for i, t in enumerate(tokens) if t not in (0, 1)]
        for i in non_special:
            self.assertEqual(labels[i], tokens[i])


class TestMLMLoss(unittest.TestCase):
    def test_loss_perfecta(self):
        # Si logits.argmax == label, loss = -log(softmax_max) ~ bajo
        rng = np.random.default_rng(0)
        vocab = 10
        n = 5
        labels = np.array([0, 1, 2, 3, 4])
        logits = np.zeros((n, vocab))
        for i, l in enumerate(labels):
            logits[i, l] = 10  # alta confianza en el label
        loss = main.compute_mlm_loss(logits, labels)
        self.assertLess(loss, 0.01)

    def test_loss_alta_con_predicciones_random(self):
        # Si logits uniformes, loss = log(vocab)
        rng = np.random.default_rng(0)
        vocab = 100
        n = 5
        labels = np.array([0, 1, 2, 3, 4])
        logits = np.zeros((n, vocab))
        loss = main.compute_mlm_loss(logits, labels)
        # log(100) ~ 4.6
        self.assertGreater(loss, 4.0)
        self.assertLess(loss, 5.0)

    def test_ignora_mascara(self):
        # Si todas las labels son -100, loss = 0
        logits = np.random.default_rng(0).standard_normal((5, 10))
        labels = np.full(5, -100)
        loss = main.compute_mlm_loss(logits, labels)
        self.assertEqual(loss, 0.0)


class TestNSP(unittest.TestCase):
    def test_label_is_next(self):
        _, pair = main.create_nsp_labels([1, 2], [3, 4], is_next=True)
        self.assertEqual(pair[0], [1, 2])
        self.assertEqual(pair[1], [3, 4])

    def test_label_not_next(self):
        label, _ = main.create_nsp_labels([1, 2], [5, 6], is_next=False)
        self.assertEqual(label, 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("MLM loss", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()