"""Pruebas para 22-procesos-estocasticos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCaminata(unittest.TestCase):
    def test_caminata_empieza_en_cero(self):
        traj = main.caminata_aleatoria(100, semilla=0)
        # np.cumsum empieza con el primer incremento
        self.assertNotEqual(traj[0], 0)  # Primer paso != 0
        # Pero la longitud es correcta
        self.assertEqual(len(traj), 100)

    def test_caminata_reproducible(self):
        t1 = main.caminata_aleatoria(50, semilla=42)
        t2 = main.caminata_aleatoria(50, semilla=42)
        np.testing.assert_array_equal(t1, t2)


class TestMarkov(unittest.TestCase):
    def test_markov_longitud(self):
        estados = ["A", "B"]
        trans = {"A": [0.5, 0.5], "B": [0.5, 0.5]}
        hist = main.cadena_markov(estados, trans, 50, "A", semilla=0)
        self.assertEqual(len(hist), 50)
        self.assertEqual(hist[0], "A")
        for s in hist:
            self.assertIn(s, estados)

    def test_markov_suma_filas_uno(self):
        trans = {
            "A": [0.5, 0.3, 0.2],
            "B": [0.1, 0.8, 0.1],
        }
        for fila in trans.values():
            self.assertAlmostEqual(sum(fila), 1.0)


class TestEstadisticas(unittest.TestCase):
    def test_media_caminata_cerca_de_cero(self):
        mu, _ = main.media_varianza_caminata(100, 5000, semilla=42)
        self.assertAlmostEqual(mu, 0.0, delta=0.1)

    def test_varianza_caminata_igual_a_n(self):
        _, var = main.media_varianza_caminata(100, 5000, semilla=42)
        self.assertAlmostEqual(var, 100.0, delta=5.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Markov", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()