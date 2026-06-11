"""Pruebas para 16-evaluacion-de-modelos-generativos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestElo(unittest.TestCase):
    def test_a_gana_sube(self):
        a, b = main.elo_rating(1500, 1500, score_a=1.0)
        self.assertGreater(a, 1500)
        self.assertLess(b, 1500)

    def test_b_gana_sube_b(self):
        a, b = main.elo_rating(1500, 1500, score_a=0.0)
        self.assertLess(a, 1500)
        self.assertGreater(b, 1500)

    def test_tie(self):
        a, b = main.elo_rating(1500, 1500, score_a=0.5)
        # Empate: ratings casi iguales
        self.assertAlmostEqual(a, 1500, places=0)
        self.assertAlmostEqual(b, 1500, places=0)


class TestMOS(unittest.TestCase):
    def test_mos(self):
        scores = np.array([4, 5, 3, 5, 4])
        self.assertAlmostEqual(main.mos_score(scores), 4.2, places=4)


class TestPValue(unittest.TestCase):
    def test_significativo_si_muchos_wins(self):
        p = main.pair_preference_pvalue(wins_a=80, wins_b=20)
        # Muy significativo
        self.assertLess(p, 0.001)

    def test_no_significativo_si_50_50(self):
        p = main.pair_preference_pvalue(wins_a=50, wins_b=50)
        # z=0, p=0.5 exactamente
        self.assertGreaterEqual(p, 0.49)
        self.assertLessEqual(p, 0.51)


class TestBenchmark(unittest.TestCase):
    def test_ocho_benchmarks(self):
        s = main.benchmark_summary()
        self.assertEqual(len(s), 8)


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