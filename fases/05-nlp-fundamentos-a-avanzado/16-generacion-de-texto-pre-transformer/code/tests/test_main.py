"""Pruebas para 16-generacion-de-texto-pre-transformer."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGreedy(unittest.TestCase):
    def test_basico(self):
        self.assertEqual(main.greedy_decode(np.array([0.1, 0.5, 0.3])), 1)


class TestTemperature(unittest.TestCase):
    def test_baja_mas_sharp(self):
        logits = np.array([2.0, 1.0, 0.5])
        p_normal = main.softmax(logits)
        p_baja = main.softmax(main.temperature_scale(logits, 0.1))
        # T baja -> mas peaky
        self.assertGreater(p_baja[0], p_normal[0])


class TestTopK(unittest.TestCase):
    def test_filtra(self):
        logits = np.array([1.0, 2.0, 0.5, 0.1])
        filtered = main.top_k_filter(logits, 2)
        # Solo el top-2 (2.0 y 1.0) sobreviven
        # El resto deberia ser -inf
        self.assertEqual(int(np.argmax(filtered)), 1)


class TestNucleus(unittest.TestCase):
    def test_suma_uno(self):
        logits = np.array([1.0, 2.0, 0.5, 0.1, -1.0])
        probs = main.nucleus_filter(logits, 0.9)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


class TestBeam(unittest.TestCase):
    def test_beam(self):
        # T=2, V=3
        logits_seq = np.array([[1.0, 0.5, 0.1], [0.1, 2.0, 0.5]])
        best = main.beam_search(logits_seq, beam_width=2)
        self.assertEqual(len(best), 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Greedy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()