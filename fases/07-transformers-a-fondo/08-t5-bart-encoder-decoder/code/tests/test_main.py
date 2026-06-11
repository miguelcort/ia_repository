"""Pruebas para 08-t5-bart-encoder-decoder."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSpanCorrupt(unittest.TestCase):
    def test_shape(self):
        tokens = list(range(50))
        inp, tgt = main.span_corrupt(tokens, mask_id=99, seed=0)
        self.assertEqual(inp.shape, (50,))
        self.assertEqual(tgt.shape, (50,))

    def test_target_coincide_con_original(self):
        # Donde tgt != -100, deberia ser el original
        tokens = list(range(20))
        inp, tgt = main.span_corrupt(tokens, mask_id=99, ratio=1.0, seed=0)
        non_ignored = np.where(tgt != -100)[0]
        for i in non_ignored:
            self.assertEqual(tgt[i], tokens[i])

    def test_input_maskeado(self):
        tokens = list(range(20))
        inp, tgt = main.span_corrupt(tokens, mask_id=99, ratio=1.0, seed=0)
        # Input deberia tener [MASK] en posiciones de mask
        non_ignored = np.where(tgt != -100)[0]
        for i in non_ignored:
            self.assertEqual(inp[i], 99)


class TestShiftRight(unittest.TestCase):
    def test_prepend_decoder_start(self):
        inp = np.array([5, 10, 15, 20])
        out = main.shift_right(inp, decoder_start_id=0)
        np.testing.assert_array_equal(out, [0, 5, 10, 15])

    def test_drop_last(self):
        inp = np.array([1, 2, 3, 4, 5])
        out = main.shift_right(inp, decoder_start_id=99)
        np.testing.assert_array_equal(out, [99, 1, 2, 3, 4])


class TestLabelSmoothing(unittest.TestCase):
    def test_loss_perfecta(self):
        # Logits.argmax = target -> loss muy baja
        logits = np.zeros((3, 5))
        targets = np.array([0, 1, 2])
        for i, t in enumerate(targets):
            logits[i, t] = 100
        loss = main.label_smoothing_loss(logits, targets, eps=0.0)
        self.assertLess(loss, 0.01)

    def test_ignora_padding(self):
        logits = np.random.default_rng(0).standard_normal((5, 10))
        targets = np.full(5, -100)
        loss = main.label_smoothing_loss(logits, targets)
        self.assertEqual(loss, 0.0)


class TestBARTNoise(unittest.TestCase):
    def test_smaller_or_equal(self):
        tokens = list(range(20))
        noisy = main.bart_noise(tokens, mask_id=99, p_mask=0.3, p_delete=0.3, seed=0)
        # Puede ser mas corto por deletes
        self.assertLessEqual(len(noisy), len(tokens))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()